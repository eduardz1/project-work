# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import operator


class CheckRange(argparse.Action):
    ops = {
        "inf": operator.gt,
        "min": operator.ge,
        "sup": operator.lt,
        "max": operator.le,
    }

    def __init__(self, *args, **kwargs):
        if "min" in kwargs and "inf" in kwargs:
            raise ValueError("either min or inf, but not both")
        if "max" in kwargs and "sup" in kwargs:
            raise ValueError("either max or sup, but not both")

        for name in self.ops:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))

        super().__init__(*args, **kwargs)

    def interval(self):
        if hasattr(self, "min"):
            lower = f"[{self.min}"
        elif hasattr(self, "inf"):
            lower = f"({self.inf}"
        else:
            lower = "(-infinity"

        if hasattr(self, "max"):
            upper = f"{self.max}]"
        elif hasattr(self, "sup"):
            upper = f"{self.sup})"
        else:
            upper = "+infinity)"

        return f"valid range: {lower}, {upper}"

    def __call__(self, parser, namespace, values, option_string=None):
        values_to_check = values if isinstance(values, list) else [values]

        for value in values_to_check:
            for name, op in self.ops.items():
                if hasattr(self, name) and not op(value, getattr(self, name)):
                    raise argparse.ArgumentError(self, self.interval())

        setattr(namespace, self.dest, values)
