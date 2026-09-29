"""Course 223: R-Style Column Vector Transformations & Missing Value Imputer"""
class RDataFrameVectorOps:
    @staticmethod
    def ifelse_vector(condition: list, yes_val, no_val) -> list:
        return [yes_val if c else no_val for c in condition]

    @staticmethod
    def impute_na_mean(column: list) -> list:
        valid_vals = [x for x in column if x is not None]
        mean_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        return [mean_val if x is None else x for x in column]
