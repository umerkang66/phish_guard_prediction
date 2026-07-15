import sys


class CustomException(Exception):
    """Base exception for the networksecurity package."""

    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = str(error_message)

        _, _, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            while exc_tb.tb_next:
                exc_tb = exc_tb.tb_next
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None

    def __str__(self):
        return (
            f"Error occurred in script: [{self.file_name}] "
            f"at line number: [{self.lineno}]. "
            f"Error message: [{self.error_message}]"
        )
