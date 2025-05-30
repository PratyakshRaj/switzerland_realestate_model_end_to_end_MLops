from pydantic import BaseModel

class realestate(BaseModel):
    type : str
    amenities : str
    postcode : int
    Area : str
    living_area : float 
    surface_of_garden_for_houses : float 	
    Building_height	: float
    proprerty_average_height : float	
    number_of_floors : int	
    perimeter : float	
    age_in_years : float