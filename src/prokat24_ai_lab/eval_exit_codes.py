def get_eval_exit_code(failed_count: int) -> int:
    if failed_count == 0:
        return 0

    return 1
