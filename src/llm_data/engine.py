from dataclasses import dataclass, field

from daft import DataFrame


@dataclass
class DataEngine:
    components: list = field(default_factory=list)

    name: str = "DataEngine"

    def add(self, name):
        self.components.append(name)
        return self

    def run(self, df: DataFrame = None) -> DataFrame:
        for component in self.components:
            df = component(df)
        return df

    def __repr__(self) -> str:
        components = ", ".join(getattr(c, "name") for c in self.components)
        return f"DataEngine(name={self.name!r}, components=[{components}])"
