
from typing import Any, Generic, Literal

from pydantic import BaseModel, PrivateAttr, TypeAdapter, model_serializer, model_validator
from pydantic._internal._forward_ref import PydanticRecursiveRef
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema
from typing_extensions import TypeVar


class MyModel(BaseModel):
    a: int

T = TypeVar("T", bound=BaseModel)

class List(BaseModel, Generic[T]): 
  _list: list[T] = PrivateAttr(default_factory=list[T])
  _adapter: TypeAdapter[list[T]] = PrivateAttr(default_factory=lambda:TypeAdapter(list[T]))

  def __init__(self, items: list[T]):
    super().__init__(_list=items)
    self._list = items

  @model_serializer(when_used="always")
  def serialize(self) -> list[T]:
    return self._list

  @model_validator(mode="before")
  @classmethod
  def validate(cls, value: list[T]) -> list[T]:
    return {"_list": value}

  @classmethod
  def model_validate(cls, obj: Any, strict: bool = False, from_attributes: bool = False, context: dict[str, Any] = ...) -> T:
    return cls(cls._adapter.validate_python(obj, strict=strict, from_attributes=from_attributes, context=context))

  @classmethod
  def model_json_schema(cls, by_alias: bool = False) -> dict[str, Any]:
    return cls._adapter.json_schema(by_alias=by_alias)

  @classmethod
  def model_validate_json(cls, json_data: str | bytes | bytearray, *, strict: bool | None = None, context: Any | None = None) -> "List":
    return cls(cls._adapter.validate_json(json_data, strict=strict, context=context))
  @classmethod
  def __class_getitem__(cls, typevar_values: type[Any] | tuple[type[Any], ...]) -> BaseModel | PydanticRecursiveRef:
    cls._adapter = TypeAdapter(list[typevar_values])
    return cls

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self._list})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({self._list})"


print(List[MyModel]([MyModel(a=1)]))
print(List[MyModel].model_json_schema())
print(List[MyModel].model_validate([{"a": 1}]).model_dump())
print(List[MyModel](items=[MyModel(a=1).model_dump()]))
print(List[MyModel].model_validate_json('[{"a": 1}]'))


T = TypeAdapter(list[MyModel])

# print(T.validate_python([{'a': 1}]))
json = T.dump_json([MyModel(a=1)]).decode()
print(json)

print(T.validate_json(json))
