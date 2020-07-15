# Copyright 2020 Google LLC. All Rights Reserved.
#
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
"""TFX DataViewBinder component executor."""

from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_executor


# Keys for input_dict.
INPUT_EXAMPLES_KEY = 'input_examples'
DATA_VIEW_KEY = 'data_view'

# Keys for output_dict.
OUTPUT_EXAMPLES_KEY = 'output_examples'

# Custom properties attached to the output Examples artifacts
DATA_VIEW_ID_PROPERTY_KEY = 'data_view_id'
DATA_VIEW_URI_PROPERTY_KEY = 'data_view_uri'


class DataViewBinderExecutor(base_executor.BaseExecutor):
  """Executor for DataViewBinder."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    del exec_properties

    # If multiple DataViews are available, use the latest one.
    latest_data_view_artifact = max(
        input_dict.get(DATA_VIEW_KEY), key=lambda a: a.id)

    input_examples = input_dict.get(INPUT_EXAMPLES_KEY, [])
    output_examples = output_dict.get(OUTPUT_EXAMPLES_KEY, [])
    assert len(input_examples) == len(output_examples), (
        'Expected the input and output Examples channel to contain the same '
        'number of Examples artifacts, but got {} and {}.'.format(
            len(input_examples), len(output_examples)))

    for ia, oa in zip(input_examples, output_examples):
      oa.uri = ia.uri
      oa.copy_custom_properties_from(ia)
      oa.set_int_custom_property(DATA_VIEW_ID_PROPERTY_KEY,
                                 latest_data_view_artifact.id)
      oa.set_string_custom_property(DATA_VIEW_URI_PROPERTY_KEY,
                                    latest_data_view_artifact.uri)
