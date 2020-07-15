# Lint as: python2, python3
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
"""Tests for tfx.components.data_view.binder_executor."""
import tensorflow as tf
from tfx.components.data_view import binder_executor
from tfx.types import standard_artifacts


class BinderExecutorTest(tf.test.TestCase):

  def testDo(self):
    data_view1 = standard_artifacts.DataView()
    data_view1.uri = '/old/data_view'
    data_view1.id = 1

    data_view2 = standard_artifacts.DataView()
    data_view2.uri = '/new/data_view'
    data_view2.id = 2

    existing_custom_property = 'payload_format'
    input_examples1 = standard_artifacts.Examples()
    input_examples1.uri = '/examples/1'
    input_examples1.set_string_custom_property(existing_custom_property,
                                               'VALUE1')

    input_examples2 = standard_artifacts.Examples()
    input_examples2.uri = '/examples/2'
    input_examples2.set_string_custom_property(existing_custom_property,
                                               'VALUE1')
    input_dict = {
        binder_executor.INPUT_EXAMPLES_KEY: [input_examples1, input_examples2],
        binder_executor.DATA_VIEW_KEY: [data_view1, data_view2],
    }
    exec_properties = {}
    output_dict = {
        binder_executor.OUTPUT_EXAMPLES_KEY: [
            standard_artifacts.Examples(),
            standard_artifacts.Examples()
        ]
    }

    executor = binder_executor.DataViewBinderExecutor()
    executor.Do(input_dict, output_dict, exec_properties)

    output_examples = output_dict.get(binder_executor.OUTPUT_EXAMPLES_KEY)
    self.assertIsNotNone(output_examples)
    self.assertLen(output_examples, 2)

    # data_view2 is newer, so its info should be bound to output examples.
    for oe in output_examples:
      self.assertEqual(
          oe.get_string_custom_property(
              binder_executor.DATA_VIEW_URI_PROPERTY_KEY), data_view2.uri)
      self.assertEqual(
          oe.get_int_custom_property(
              binder_executor.DATA_VIEW_ID_PROPERTY_KEY), data_view2.id)

    for oe, ie in zip(output_examples, [input_examples1, input_examples2]):
      # output should share the URI with the input.
      self.assertEqual(oe.uri, ie.uri)
      # other custom properties should be inherited.
      self.assertEqual(
          oe.get_string_custom_property(existing_custom_property),
          ie.get_string_custom_property(existing_custom_property))


if __name__ == '__main__':
  tf.test.main()
