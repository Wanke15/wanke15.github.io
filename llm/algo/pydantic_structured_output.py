# 格式化输出
from enum import Enum
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class MakeCls(Enum):
    Toyota = "Toyota"
    Honda = "Honda"
    Ford = "Ford"
    Suzuki = "Suzuki"

class Vehicle(BaseModel):
    Type: str = Field(
        ...,
        examples=["Car", "Truck", "Motorcycle", 'Bus'],
        description="Return the type of the vehicle.",
    )
    License: str = Field(
        ...,
        description="Return the license plate number of the vehicle.",
    )
    # （1）普通举例
    # Make: str = Field(
    #     ...,
    #     examples=["Toyota", "Honda", "Ford", "Suzuki"],
    #     description="Return the Make of the vehicle.",
    # )
    # （2）自定义枚举类
    # Make: MakeCls = Field(
    #     ...,
    #     description="Return the Make of the vehicle.",
    # )
    # （3）Literal类型枚举型字段
    Make: Literal["Toyota", "Honda", "Ford", "Suzuki"] = Field(
        ...,
        description="Return the Make of the vehicle.",
    )

    Model: str = Field(
        ...,
        examples=["Corolla", "Civic", "F-150"],
        description="Return the Model of the vehicle.",
    )
    Color: str = Field(
        ...,
        example=["Red", "Blue", "Black", "White"],
        description="Return the color of the vehicle.",
    )

parser = JsonOutputParser(pydantic_object=Vehicle)
instructions = parser.get_format_instructions()
print(instructions)

"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"Type": {"description": "Return the type of the vehicle.", "examples": ["Car", "Truck", "Motorcycle", "Bus"], "title": "Type", "type": "string"}, "License": {"description": "Return the license plate number of the vehicle.", "title": "License", "type": "string"}, "Make": {"description": "Return the Make of the vehicle.", "enum": ["Toyota", "Honda", "Ford", "Suzuki"], "title": "Make", "type": "string"}, "Model": {"description": "Return the Model of the vehicle.", "examples": ["Corolla", "Civic", "F-150"], "title": "Model", "type": "string"}, "Color": {"description": "Return the color of the vehicle.", "example": ["Red", "Blue", "Black", "White"], "title": "Color", "type": "string"}}, "required": ["Type", "License", "Make", "Model", "Color"]}
```

"""
