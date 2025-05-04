from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class SocialStatus(str, Enum):
    upper_class = "upper class"
    middle_class = "middle class"
    lower_class = "lower class"

class Related_Character(BaseModel):
    character_id: str = Field(description="The unique identifier for the character that is connected.")

class Character(BaseModel):
    character_id: str = Field(description="The unique identifier for the character. Use a unique combination of name and number.")
    name: str = Field(description="The actual name of the character in the book. Don't use pronouns unless the character is not named.")
    social_status: SocialStatus = Field(description="The social status of the character in the social hierarchy")
    connections: List[Related_Character] = Field(description="The characters that are connected to this character. The connections are only based on the interactions in the book.")
    
class Social_Network(BaseModel):
    characters: List[Character] = Field(description="The characters in the social network")