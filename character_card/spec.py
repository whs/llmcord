import base64
import re
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
import pydantic
import png


def load_card(b: bytes) -> Union["TavernCard", None]:
    reader = png.Reader(bytes=b)
    for chunk_type, content in reader.chunks():
        if chunk_type == b"tEXt":
            prefix = b"chara\00"
            if content.startswith(prefix):
                json_content = base64.b64decode(content[len(prefix):])
                try:
                    return TavernCard.model_validate_json(json_content)
                except pydantic.ValidationError:
                    continue

class CharacterBookEntry(pydantic.BaseModel):
    keys: List[str]
    content: str
    extensions: Dict[str, Any]
    enabled: bool
    insertion_order: int
    case_sensitive: Optional[bool] = None
    name: Optional[str] = None
    priority: Optional[int] = None
    id: Optional[int] = None
    comment: Optional[str] = None
    selective: Optional[bool] = None
    secondary_keys: Optional[List[str]] = None
    constant: Optional[bool] = None
    position: Optional[Literal['before_char', 'after_char']] = None


class CharacterBook(pydantic.BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    scan_depth: Optional[int] = None
    token_budget: Optional[int] = None
    recursive_scanning: Optional[bool] = None
    extensions: Dict[str, Any]
    entries: List[CharacterBookEntry]


class TavernCardV1(pydantic.BaseModel):
    name: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str

    def to_v2(self) -> "TavernCardV2":
        v2_data = TavernCardV2Data(
            name=self.name,
            description=self.description,
            personality=self.personality,
            scenario=self.scenario,
            first_mes=self.first_mes,
            mes_example=self.mes_example,
            creator_notes="",
            system_prompt="",
            post_history_instructions="",
            alternate_greetings=[],
            tags=[],
            creator="",
            character_version="0.0.0",
        )
        
        return TavernCardV2(data=v2_data)

char_re = re.compile(r"~\{\{char}}|<bot>~", re.I)
user_re = re.compile(r"~\{\{user}}|<user>~", re.I)
original_re = re.compile(r"~\{\{original}}~", re.I)

class TavernCardV2Data(pydantic.BaseModel):
    name: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str
    creator_notes: str
    system_prompt: str
    post_history_instructions: str
    alternate_greetings: List[str]
    character_book: Optional[CharacterBook] = None
    tags: List[str]
    creator: str
    character_version: str
    extensions: Dict[str, Any] = {}

    def templatize(self, template: str, username: str, original_prompt="") -> str:
        template = char_re.sub(self.name, template)
        template = user_re.sub(username, template)
        template = original_re.sub(original_prompt, template)

        return template


class TavernCardV2(pydantic.BaseModel):
    spec: Literal['chara_card_v2'] = 'chara_card_v2'
    spec_version: Literal['2.0'] = '2.0'
    data: TavernCardV2Data

def tavern_card_version_discriminator(v: Any) -> str:
    if "spec" in v:
        return v["spec"]

    return "chara_card_v1"


class TavernCard(pydantic.RootModel):
    root: Annotated[
        Union[
            Annotated[TavernCardV1, pydantic.Tag("chara_card_v1")],
            Annotated[TavernCardV2, pydantic.Tag("chara_card_v2")]
        ],
        pydantic.Discriminator(tavern_card_version_discriminator)
    ]
