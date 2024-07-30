import re
from typing import List, Union

import numpy as np

###################################################################### SPECIAL TOKENS

CON_TOKEN = "<con>"  # Change-of-note (change-of-note) token
COC_TOKEN = "<coc>"  # Change-of-column (change-of-voice) token
COR_TOKEN = "<cor>"  # Change-of-row (change-of-event) token

###################################################################### KRN PARSER

ENCODING_OPTIONS = ["kern", "bekern", "ekern"]


class krnParser:
    """Main Kern parser operations class."""

    def __init__(self, encoding: str = "bekern") -> None:
        # Check encoding
        assert (
            encoding in ENCODING_OPTIONS
        ), f"You must chose one of the possible encoding options: {','.join(ENCODING_OPTIONS)}"

        # Attributes
        self.encoding = encoding
        if encoding == "kern":
            self.header_word = "**kern"
        elif encoding == "bekern":
            self.header_word = "**bekern"
        elif encoding == "ekern":
            self.header_word = "**ekern"
        self.reserved_words = ["clef", "*k[", "*M"]
        self.reserved_dot = "."
        self.reserved_dot_EncodedCharacter = "DOT"
        self.clef_change_other_voices = "*"
        self.open_spine = "*^"
        self.close_spine = "*v"
        self.comment_symbols = ["*", "!"]

    # ---------------------------------------------------------------------------- AUXILIARY FUNCTIONS

    def _readSrcFile(self, file_path: str) -> np.ndarray:
        """Read polyphonic kern file and adequate the format for further processes."""
        with open(file_path) as fin:
            in_src = fin.read().splitlines()
        return np.array(in_src)

    def _postprocessKernSequence(self, in_score: list) -> list:
        """Exchanging '*' for the actual symbol."""

        # Retrieving positions with '*'
        positions_elements = list(
            np.where(
                np.array(
                    [
                        ("*" in line)
                        and (self.open_spine not in line)
                        and (self.close_spine not in line)
                        for line in in_score
                    ]
                )
                is True
            )[0]
        )
        positions_voices = [
            np.where(np.array(in_score[idx]) == "*")[0] for idx in positions_elements
        ]

        # For each position, we retrieve the last explicit clef symbol and include it in the stream
        for it_position in range(len(positions_elements)):
            position_element = positions_elements[it_position]
            for position_voice in positions_voices[it_position]:
                # Locating last position with the required number of voices (in case spines open/close)
                it_reference = position_element
                end_search = False
                while it_reference >= 0 and end_search is False:
                    if len(in_score[it_reference]) >= position_voice + 1:
                        it_reference -= 1
                    else:
                        it_reference += 1
                        end_search = True
                    pass
                pass

                previous_elements = [
                    line[position_voice]
                    for line in in_score[it_reference:position_element]
                ]
                try:
                    new_element = in_score[
                        it_reference
                        + max(np.where(np.char.startswith(previous_elements, "*clef")))[
                            0
                        ]
                    ][position_voice]
                except Exception:
                    new_element = in_score[position_element][position_voice - 1]
                in_score[position_element][position_voice] = new_element
            pass
        pass

        return in_score

    def _cleanKernFile(self, file_path: str) -> list:
        """Convert complete kern sequence to CLEAN kern format."""
        in_file = self._readSrcFile(file_path=file_path)

        # Processing individual voices
        out_score = []
        for step in in_file:
            # Splitting voices
            voices = step.split("\t")

            # Iterating through voices
            current_step = []
            any_voice_empty = False
            for single_voice in voices:
                try:
                    tokens_voice = [self._cleanKernToken(u) for u in single_voice.split(" ")]
                    # if len(tokens_voice) == 0:
                    #     any_voice_empty = True
                    #     break
                    current_step.append(
                        " ".join(tokens_voice
                        )
                    )
                except Exception as e:
                    # print(e)
                    # # self._cleanKernToken(u) is None
                    # any_voice_empty = True
                    # break
                    pass
            # Control if each voice has at least one token 
            if any_voice_empty:
                # if "**kern" not in voices and "*-" not in voices and "staff" not in voices and "*met(" not in voices:
                #     print("Any token out from", voices, "converted wrong", current_step)
                pass
            elif len(current_step) > 0:
                out_score.append(current_step)
        pass

        # Postprocess obtained score
        out_score = self._postprocessKernSequence(out_score)

        return out_score

    def _cleanKernToken(self, in_token: str) -> Union[str, None]:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default
        in_token = in_token.replace("·", "")  # Remove dot separator in bekern
           

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token = in_token

        elif in_token == self.reserved_dot:  # Case when using '.' for sync. voices
            out_token = self.reserved_dot_EncodedCharacter

        elif (
            in_token.strip() == self.clef_change_other_voices
        ):  # Clef change in other voices
            out_token = in_token

        elif (
            in_token.strip() == self.open_spine or in_token.strip() == self.close_spine
        ):  # Open/close spine
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token = None

        elif in_token.startswith("s"):  # Slurs
            out_token = "s"

        elif "=" in in_token:  # Bar lines
            out_token = "="

        elif "q" not in in_token:
            if "rr" in in_token:
                out_token = re.findall("rr[0-9]+", in_token)[0]  # Multirest
            elif "r" in in_token:
                out_token = in_token.split("r")[0] + "r"  # Rest
            else:
                out_token = re.findall("\d+[.]*[a-gA-G]+[n#-]*", in_token)[
                    0
                ]  # Music note
                if "[" in in_token:
                    out_token += "["
                if "]" in in_token:
                    out_token += "]"

        elif "q" in in_token:
            out_token = re.findall("\d*[a-gA-G]+[n#-]*[q]+", in_token)[
                0
            ]  # Music note with q

        return out_token

    # ---------------------------------------------------------------------------- ENCODE CALL

    def encode(self, file_path: str) -> List[str]:
        y_clean = self._cleanKernFile(file_path)
        y_coded = []

        for i, voices in enumerate(y_clean):
            for j, voice in enumerate(voices):
                notes = voice.split()
                for k, note in enumerate(notes):
                    y_coded.append(note)
                    if k != len(notes) - 1:
                        y_coded.append(CON_TOKEN)
                if j != len(voices) - 1:
                    y_coded.append(COC_TOKEN)
            if i != len(y_clean) - 1:
                y_coded.append(COR_TOKEN)

        return y_coded
