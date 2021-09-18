from numpy.lib.function_base import iterable
from pandas.core.frame import DataFrame

class InvalidFilePath(Exception):
    """
    Exception for invalid filepath.

    Raises an exception if string is not a valid file path.
    ...

    Attributes
    ----------
    file_path: str
        Path to file. 
    """

    def __init__(self, file_path: object) -> None:
        self.file_path = file_path

    def __str__(self):
        return "{} is not a valid file path.".format(self.file_path)

class InvalidDataFrame(Exception):
    """
    Exception for invalid DataFrame.

    Raises an exception if object is not a DataFrame.
    ...

    Attributes
    ----------
    df: DataFrame
        Object to validate if it is a DataFrame.
    """
    def __init__(self, df: DataFrame) -> None: 
        self.df = df
    
    def __str__(self):        
        return "Object supplied is not a valid DataFrame."

class UnexpectedDataFrame(Exception):
    """
    Exception for Unexpected DataFrame.

    Raises an exception if when DataFrame is validated and returns False.
    ...

    Attributes
    ---------
    df: DataFrame
        DataFrame to validate.
    """
    def __init__(self, df: DataFrame) -> None: 
        self.df = df
    
    def __str__(self):
        return "DataFrame supplied is not expected."

class InvalidBeverage(Exception):
    """
    Exception for Invalid Beverage.

    Raises an exception if beverage is not in list of beverages.
    ...

    Attributes
    ----------
    *bevs: iterable
        List of beverages.
    """
    def __init__(self, *bevs: list) -> None:
        self.bevs = bevs
        
    def __str__(self):
        return "{} not in list of beverages.".format(self.bevs)


class InvalidRegion(Exception):
    """
    Exception for Invalid Region.

    Raises an exception if region is not in list of regions.
    ...

    Attributes
    ----------
    region: str
        String to check if in list of regions.
    """
    def __init__(self, region: object) -> None:
        self.region = region

    def __str__(self):
        return "{} is not a region in regions.".format(self.region)

class BeverageRegionExceptions(Exception):
    """
    Exception for both Invalid Beverage and Invalid Region.

    Raises an exception if beverage and region supplied are not in list of beverages and list of regions, respectively.
    ...

    Attributes
    ----------
    beverage: str
        String to check if in list of beverages.
    
    region: str
        String to check if in list of regions.
    """
    def __init__(self, beverage, region):
        self.beverage = beverage
        self.region = region
    
    def __str__(self):
        return "{} not found in beverage list and, {} not found in regions.".format(self.beverage, self.region)

        