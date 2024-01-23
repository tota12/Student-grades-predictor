import sys
import logging


def error_msg_details(error, error_detail: sys):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg_details = (
        f'error occured in {file_name} at line {line_number}, message: {error}'
    )

    return error_msg_details


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
