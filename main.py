# Copyright © 2025 Cognizant Technology Solutions Corp, www.cognizant.com.
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
#
# END COPYRIGHT

from dotenv import load_dotenv

from core.experiment.cli import parse_args
from core.experiment.config import build_config
from core.experiment.runner import SimulationRunner

load_dotenv()


def main():
    args = parse_args()
    params = build_config(args)

    resume = args.resume
    runner = SimulationRunner(params=params, resume=resume)
    runner.run()


if __name__ == "__main__":
    main()
