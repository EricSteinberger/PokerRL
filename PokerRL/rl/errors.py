# Copyright (c) 2019 Eric Steinberger


class UnknownModeError(ValueError):

    def __init__(self, var):
        print("Mode", var, "is unknown")
