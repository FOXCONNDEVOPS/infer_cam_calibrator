import numpy as np
from typing import Any
import json
from dataclasses import dataclass
from models.coord import Coord

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Coord):
            return obj.to_json() 
        else:
            return super().default(obj)

@dataclass
class Box:
    coord: Coord
    confidence: float
    distance: int
    class_id: int
    class_name: str
    original_size: tuple[int, int] # widht, height
    cam_idx: int # 0 is rgb and 1 is nir

    def serialize(self) -> dict[str, Any]:
        return {
            "coord": self.coord.serialize(),
            "confidence": self.confidence,
            "distance": self.distance,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "original_size": self.original_size,
            "cam_idx": self.cam_idx,
        }
    
    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> 'Box':
        return cls(
            coord=Coord.deserialize(d["coord"]),
            confidence=d["confidence"],
            distance=d["distance"],
            class_id=d["class_id"],
            class_name=d["class_name"],
            original_size=d["original_size"],
            cam_idx=d["cam_idx"],
        )

    
def sort_boxes(boxes: list[Box], class_name: str) -> list[Box]:
    column: list[Box] = []
    for box in boxes:
        if box.class_name == class_name:
            column.append(box)
    column.sort(key=lambda box: box.coord.bl[1])
    return column


def bar_center(boxes: list[Box]) -> float:
    bars: list[Box] = []
    for box in boxes:
        if box.class_name == 'bar':
            bars.append(box)
    centers = []
    for bar in bars:
        centers.append((bar.coord.bl[1]+bar.coord.tr[1])/2)
    if len(centers) == 0:
        raise ValueError(f"Coudn't find a bar for this image {box.distance}")
    return float(np.mean(centers))

