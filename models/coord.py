from dataclasses import dataclass
from typing import Any

@dataclass
class Coord:
    bl: tuple[int, int] # x,y
    br: tuple[int, int]
    tl: tuple[int, int]
    tr: tuple[int, int]

    def serialize(self) -> dict[str, Any]:
        return {
            "bl": self.bl,
            "br": self.br,
            "tl": self.tl,
            "tr": self.tr,
        }
        
    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> 'Coord':
        return cls(
            bl=d["bl"],
            br=d["br"],
            tl=d["tl"],
            tr=d["tr"],
        )