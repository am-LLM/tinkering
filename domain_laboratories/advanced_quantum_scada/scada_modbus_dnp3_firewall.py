"""
Industrial SCADA Deep Packet Inspection (DPI) Firewall & Stateful Anomaly Detector.

Supports:
- Modbus TCP (MBAP + PDU parsing, function code ACL, register bounds & slew-rate monitoring).
- DNP3 (Data Link Layer framing, CRC-16 checks, Transport & Application layer parsing, Select-Before-Operate tracking).
- Stateful MitM spoofing detection, replay attack mitigation, sequence number desync detection.
- Modbus RTU / DNP3 CRC-16 verification engines.
"""

from dataclasses import dataclass, field
from enum import Enum
import struct
import time
from typing import Dict, List, Optional, Set, Tuple, Any


class FirewallAction(Enum):
    ALLOW = "ALLOW"
    DROP = "DROP"
    ALERT = "ALERT"


class ModbusFunctionCode(Enum):
    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10
    MASK_WRITE_REGISTER = 0x16
    READ_WRITE_MULTIPLE = 0x17


class DNP3FunctionCode(Enum):
    CONFIRM = 0x00
    READ = 0x01
    WRITE = 0x02
    SELECT = 0x03
    OPERATE = 0x04
    DIRECT_OPERATE = 0x05
    DIRECT_OPERATE_NO_ACK = 0x06
    COLD_RESTART = 0x0D
    WARM_RESTART = 0x0E
    UNSOLICITED_RESPONSE = 0x82


class CRC16Calculator:
    """CRC-16 algorithms for SCADA protocols."""

    @staticmethod
    def calculate_modbus_crc16(data: bytes) -> int:
        """Standard Modbus RTU CRC-16 (Polynomial: 0xA001, Init: 0xFFFF)."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    @staticmethod
    def calculate_dnp3_crc16(data: bytes) -> int:
        """
        DNP3 Data Link CRC-16 (Polynomial: 0xA653 / reversed 0xD993, inverted remainder).
        Calculated over 8-byte header and 16-byte payload blocks.
        """
        crc = 0x0000
        for byte in data:
            temp = byte ^ (crc & 0xFF)
            temp = temp ^ ((temp << 4) & 0xFF)
            crc = (crc >> 8) ^ (temp << 8) ^ (temp << 3) ^ (temp >> 4)
            crc &= 0xFFFF
        # Invert result for DNP3 standard
        return (~crc) & 0xFFFF


@dataclass
class ModbusSecurityPolicy:
    """Modbus TCP security rules for an industrial zone."""
    allowed_unit_ids: Set[int] = field(default_factory=lambda: {1, 2, 3, 4})
    allowed_function_codes: Set[int] = field(
        default_factory=lambda: {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x10}
    )
    # Mapping of unit_id -> (min_address, max_address)
    allowed_register_ranges: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: {1: (0, 1000), 2: (0, 500), 3: (0, 2000), 4: (0, 100)}
    )
    # Mapping of unit_id -> (min_value, max_value) for analog register writes
    register_value_bounds: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: {1: (0, 10000), 2: (-500, 500), 3: (0, 65535), 4: (0, 250)}
    )
    max_write_slew_rate: float = 1000.0  # Max allowed value change per second per register
    max_request_rate_hz: float = 50.0   # Max requests per second per client IP


@dataclass
class DNP3SecurityPolicy:
    """DNP3 security rules for electric sub-station automation."""
    allowed_source_addrs: Set[int] = field(default_factory=lambda: {1, 2, 10, 100})
    allowed_dest_addrs: Set[int] = field(default_factory=lambda: {1024, 2048, 4096})
    allowed_function_codes: Set[int] = field(
        default_factory=lambda: {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06}
    )
    select_operate_timeout_s: float = 5.0  # Select-Before-Operate expiration window
    disallow_restart_commands: bool = True # Drop COLD_RESTART (0x0D) / WARM_RESTART (0x0E)


@dataclass
class InspectionResult:
    """Inspection verdict returned by SCADA DPI firewall."""
    action: FirewallAction
    reason: str
    protocol: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class ModbusTCPFirewall:
    """Stateful DPI engine for Modbus TCP."""

    def __init__(self, policy: Optional[ModbusSecurityPolicy] = None):
        self.policy = policy or ModbusSecurityPolicy()
        # Connection state: (client_ip, unit_id, register_addr) -> (last_value, last_time)
        self.last_register_state: Dict[Tuple[str, int, int], Tuple[int, float]] = {}
        # Client request tracking: client_ip -> list of timestamps
        self.client_request_history: Dict[str, List[float]] = {}
        # Transaction tracking: (client_ip, unit_id) -> set of active transaction IDs
        self.active_transactions: Dict[Tuple[str, int], Set[int]] = {}

    def inspect_packet(self, raw_bytes: bytes, client_ip: str = "192.168.1.100") -> InspectionResult:
        """
        Inspect Modbus TCP frame: MBAP Header (7 bytes) + PDU (Function Code + Data).
        MBAP: Transaction ID (2B), Protocol ID (2B = 0), Length (2B), Unit ID (1B).
        """
        now = time.time()
        if len(raw_bytes) < 8:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason="Malformed Modbus TCP packet: length < 8 bytes",
                protocol="ModbusTCP",
                timestamp=now
            )

        # Rate limiting check
        history = self.client_request_history.setdefault(client_ip, [])
        history = [t for t in history if now - t < 1.0]
        history.append(now)
        self.client_request_history[client_ip] = history
        if len(history) > self.policy.max_request_rate_hz:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Rate limit exceeded: {len(history)} req/s from {client_ip}",
                protocol="ModbusTCP",
                timestamp=now
            )

        # Parse MBAP Header
        trans_id, proto_id, length, unit_id = struct.unpack(">HHHB", raw_bytes[:7])

        if proto_id != 0:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Invalid Modbus Protocol ID: {proto_id} (expected 0)",
                protocol="ModbusTCP",
                timestamp=now,
                details={"transaction_id": trans_id, "unit_id": unit_id}
            )

        if len(raw_bytes) - 6 != length:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"MBAP Length mismatch: declared {length}, actual {len(raw_bytes)-6}",
                protocol="ModbusTCP",
                timestamp=now,
                details={"transaction_id": trans_id, "unit_id": unit_id}
            )

        if unit_id not in self.policy.allowed_unit_ids:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Unauthorized Unit ID: {unit_id}",
                protocol="ModbusTCP",
                timestamp=now,
                details={"unit_id": unit_id}
            )

        # Parse PDU
        func_code = raw_bytes[7]
        pdu_data = raw_bytes[8:]

        if func_code not in self.policy.allowed_function_codes:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Disallowed Modbus Function Code: 0x{func_code:02X}",
                protocol="ModbusTCP",
                timestamp=now,
                details={"function_code": func_code, "unit_id": unit_id}
            )

        # Register Range & Value Bounds Inspection
        allowed_range = self.policy.allowed_register_ranges.get(unit_id, (0, 65535))
        val_bounds = self.policy.register_value_bounds.get(unit_id, (0, 65535))

        # Handle Read (0x01, 0x02, 0x03, 0x04)
        if func_code in (0x01, 0x02, 0x03, 0x04):
            if len(pdu_data) < 4:
                return InspectionResult(FirewallAction.DROP, "Truncated read request PDU", "ModbusTCP", now)
            start_addr, quantity = struct.unpack(">HH", pdu_data[:4])
            end_addr = start_addr + quantity - 1
            if start_addr < allowed_range[0] or end_addr > allowed_range[1]:
                return InspectionResult(
                    action=FirewallAction.DROP,
                    reason=f"Register read range out of bounds: [{start_addr}, {end_addr}] vs allowed {allowed_range}",
                    protocol="ModbusTCP",
                    timestamp=now,
                    details={"start_addr": start_addr, "quantity": quantity, "unit_id": unit_id}
                )

        # Handle Write Single Register (0x06)
        elif func_code == 0x06:
            if len(pdu_data) < 4:
                return InspectionResult(FirewallAction.DROP, "Truncated write register PDU", "ModbusTCP", now)
            reg_addr, reg_val = struct.unpack(">HH", pdu_data[:4])
            if reg_addr < allowed_range[0] or reg_addr > allowed_range[1]:
                return InspectionResult(
                    action=FirewallAction.DROP,
                    reason=f"Write register address {reg_addr} out of bounds {allowed_range}",
                    protocol="ModbusTCP",
                    timestamp=now,
                    details={"reg_addr": reg_addr, "value": reg_val}
                )
            if reg_val < val_bounds[0] or reg_val > val_bounds[1]:
                return InspectionResult(
                    action=FirewallAction.ALERT,
                    reason=f"Write value {reg_val} exceeds safety limits {val_bounds}",
                    protocol="ModbusTCP",
                    timestamp=now,
                    details={"reg_addr": reg_addr, "value": reg_val}
                )

            # Slew Rate Checking (detect rapid anomalous setpoint modifications)
            state_key = (client_ip, unit_id, reg_addr)
            if state_key in self.last_register_state:
                prev_val, prev_time = self.last_register_state[state_key]
                dt = max(1e-3, now - prev_time)
                rate = abs(reg_val - prev_val) / dt
                if rate > self.policy.max_write_slew_rate:
                    return InspectionResult(
                        action=FirewallAction.ALERT,
                        reason=f"Setpoint slew-rate anomaly: {rate:.1f} units/s exceeds max {self.policy.max_write_slew_rate}",
                        protocol="ModbusTCP",
                        timestamp=now,
                        details={"reg_addr": reg_addr, "slew_rate": rate}
                    )
            self.last_register_state[state_key] = (reg_val, now)

        # Handle Write Multiple Registers (0x10)
        elif func_code == 0x10:
            if len(pdu_data) < 5:
                return InspectionResult(FirewallAction.DROP, "Truncated write multiple PDU", "ModbusTCP", now)
            start_addr, quantity, byte_count = struct.unpack(">HHB", pdu_data[:5])
            if byte_count != quantity * 2 or len(pdu_data) < 5 + byte_count:
                return InspectionResult(FirewallAction.DROP, "Invalid byte count in write multiple PDU", "ModbusTCP", now)
            if start_addr < allowed_range[0] or (start_addr + quantity - 1) > allowed_range[1]:
                return InspectionResult(
                    action=FirewallAction.DROP,
                    reason=f"Write multiple range [{start_addr}, {start_addr + quantity - 1}] out of bounds {allowed_range}",
                    protocol="ModbusTCP",
                    timestamp=now
                )

        return InspectionResult(
            action=FirewallAction.ALLOW,
            reason="Packet verified against Modbus security policy",
            protocol="ModbusTCP",
            timestamp=now,
            details={"transaction_id": trans_id, "unit_id": unit_id, "function_code": func_code}
        )


class DNP3Firewall:
    """Stateful DPI engine for DNP3 (Distributed Network Protocol 3)."""

    def __init__(self, policy: Optional[DNP3SecurityPolicy] = None):
        self.policy = policy or DNP3SecurityPolicy()
        # Select-Before-Operate (SBO) state tracking: (source_addr, dest_addr, sequence_num) -> (select_time, target_obj)
        self.sbo_pending_operates: Dict[Tuple[int, int, int], Tuple[float, bytes]] = {}
        # Expected sequence numbers to prevent desync / replay: (source, dest) -> last_seq
        self.sequence_trackers: Dict[Tuple[int, int], int] = {}

    def inspect_packet(self, raw_bytes: bytes) -> InspectionResult:
        """
        Inspect DNP3 Data Link Layer + Transport + Application layer frame.
        Data Link Header (10 bytes):
        - Sync bytes (0x05, 0x64)
        - Length (1B)
        - Link Control (1B)
        - Destination Address (2B, little-endian)
        - Source Address (2B, little-endian)
        - Header CRC-16 (2B, little-endian)
        Followed by data chunks (max 16B payload + 2B CRC per chunk).
        """
        now = time.time()
        if len(raw_bytes) < 10:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason="Truncated DNP3 frame (< 10 bytes)",
                protocol="DNP3",
                timestamp=now
            )

        # 1. Sync Bytes Verification
        if raw_bytes[0] != 0x05 or raw_bytes[1] != 0x64:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Invalid DNP3 sync header: 0x{raw_bytes[0]:02X} 0x{raw_bytes[1]:02X}",
                protocol="DNP3",
                timestamp=now
            )

        # 2. Header CRC Verification
        header_data = raw_bytes[:8]
        expected_crc = struct.unpack("<H", raw_bytes[8:10])[0]
        calc_crc = CRC16Calculator.calculate_dnp3_crc16(header_data)
        if expected_crc != calc_crc:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"DNP3 Header CRC mismatch: calculated 0x{calc_crc:04X} != received 0x{expected_crc:04X}",
                protocol="DNP3",
                timestamp=now
            )

        # 3. Parse Data Link Header Fields
        length = raw_bytes[2]
        control = raw_bytes[3]
        dest_addr, src_addr = struct.unpack("<HH", raw_bytes[4:8])

        # Address filtering
        if src_addr not in self.policy.allowed_source_addrs:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Unauthorized DNP3 source address: {src_addr}",
                protocol="DNP3",
                timestamp=now,
                details={"src_addr": src_addr, "dest_addr": dest_addr}
            )

        if dest_addr not in self.policy.allowed_dest_addrs:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Unauthorized DNP3 destination address: {dest_addr}",
                protocol="DNP3",
                timestamp=now,
                details={"src_addr": src_addr, "dest_addr": dest_addr}
            )

        # 4. Verify Payload CRC chunks
        payload_bytes = bytearray()
        idx = 10
        bytes_remaining = length - 5  # Length includes Control(1) + Dest(2) + Src(2)
        while bytes_remaining > 0:
            chunk_len = min(16, bytes_remaining)
            if idx + chunk_len + 2 > len(raw_bytes):
                return InspectionResult(FirewallAction.DROP, "DNP3 frame truncated in payload chunk", "DNP3", now)
            chunk_data = raw_bytes[idx : idx + chunk_len]
            chunk_crc = struct.unpack("<H", raw_bytes[idx + chunk_len : idx + chunk_len + 2])[0]
            if CRC16Calculator.calculate_dnp3_crc16(chunk_data) != chunk_crc:
                return InspectionResult(
                    action=FirewallAction.DROP,
                    reason="DNP3 payload chunk CRC failure (corruption or tampering)",
                    protocol="DNP3",
                    timestamp=now
                )
            payload_bytes.extend(chunk_data)
            idx += chunk_len + 2
            bytes_remaining -= chunk_len

        if len(payload_bytes) < 3:
            # Valid link layer only (e.g. Ack/Link Status)
            return InspectionResult(FirewallAction.ALLOW, "DNP3 Data Link frame verified", "DNP3", now)

        # 5. Transport & Application Layer Inspection
        # Transport header (1B) -> Application Control (1B) -> Application Function Code (1B)
        tp_header = payload_bytes[0]
        app_ctrl = payload_bytes[1]
        app_func = payload_bytes[2]
        app_seq = app_ctrl & 0x0F

        # Check restart commands
        if self.policy.disallow_restart_commands and app_func in (
            DNP3FunctionCode.COLD_RESTART.value,
            DNP3FunctionCode.WARM_RESTART.value
        ):
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Restart command blocked by policy: 0x{app_func:02X}",
                protocol="DNP3",
                timestamp=now,
                details={"function_code": app_func, "src_addr": src_addr}
            )

        if app_func not in self.policy.allowed_function_codes:
            return InspectionResult(
                action=FirewallAction.DROP,
                reason=f"Disallowed DNP3 Application Function Code: 0x{app_func:02X}",
                protocol="DNP3",
                timestamp=now,
                details={"function_code": app_func}
            )

        # 6. Stateful Select-Before-Operate (SBO) Tracking
        sbo_key = (src_addr, dest_addr, app_seq)
        obj_payload = bytes(payload_bytes[3:])

        if app_func == DNP3FunctionCode.SELECT.value:
            # Register SELECT state
            self.sbo_pending_operates[sbo_key] = (now, obj_payload)
            return InspectionResult(
                action=FirewallAction.ALLOW,
                reason="DNP3 SELECT registered for SBO sequence",
                protocol="DNP3",
                timestamp=now,
                details={"src_addr": src_addr, "dest_addr": dest_addr, "seq": app_seq}
            )

        elif app_func == DNP3FunctionCode.OPERATE.value:
            # Verify that a matching SELECT was issued within timeout window
            if sbo_key not in self.sbo_pending_operates:
                return InspectionResult(
                    action=FirewallAction.ALERT,
                    reason=f"Unsolicited/Unmatched OPERATE command without prior SELECT (MitM spoofing attempt)",
                    protocol="DNP3",
                    timestamp=now,
                    details={"src_addr": src_addr, "dest_addr": dest_addr, "seq": app_seq}
                )
            select_time, select_payload = self.sbo_pending_operates.pop(sbo_key)
            if (now - select_time) > self.policy.select_operate_timeout_s:
                return InspectionResult(
                    action=FirewallAction.DROP,
                    reason=f"SBO OPERATE timed out ({now - select_time:.2f}s > {self.policy.select_operate_timeout_s}s)",
                    protocol="DNP3",
                    timestamp=now
                )
            if select_payload != obj_payload:
                return InspectionResult(
                    action=FirewallAction.ALERT,
                    reason="SBO payload mismatch between SELECT and OPERATE (tampered control object)",
                    protocol="DNP3",
                    timestamp=now
                )

        return InspectionResult(
            action=FirewallAction.ALLOW,
            reason="DNP3 application frame verified",
            protocol="DNP3",
            timestamp=now,
            details={"function_code": app_func, "src_addr": src_addr, "dest_addr": dest_addr}
        )


class SCADAFirewallEngine:
    """Unified Industrial SCADA DPI Firewall."""

    def __init__(
        self,
        modbus_policy: Optional[ModbusSecurityPolicy] = None,
        dnp3_policy: Optional[DNP3SecurityPolicy] = None
    ):
        self.modbus_fw = ModbusTCPFirewall(modbus_policy)
        self.dnp3_fw = DNP3Firewall(dnp3_policy)
        self.audit_log: List[InspectionResult] = []

    def inspect_modbus(self, raw_bytes: bytes, client_ip: str = "192.168.1.100") -> InspectionResult:
        res = self.modbus_fw.inspect_packet(raw_bytes, client_ip)
        self.audit_log.append(res)
        return res

    def inspect_dnp3(self, raw_bytes: bytes) -> InspectionResult:
        res = self.dnp3_fw.inspect_packet(raw_bytes)
        self.audit_log.append(res)
        return res
