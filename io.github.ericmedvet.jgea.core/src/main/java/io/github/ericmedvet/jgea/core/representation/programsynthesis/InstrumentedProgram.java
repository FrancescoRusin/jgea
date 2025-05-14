/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jgea.core.representation.programsynthesis;

import java.util.List;

public interface InstrumentedProgram extends Program {

  interface InstrumentedOutcome extends Program.Outcome {
    RunProfile profile();

    static InstrumentedOutcome from(List<Object> outputs, ProgramExecutionException exception, RunProfile profile) {
      record HardInstrumentedOutcome(
          List<Object> outputs, ProgramExecutionException exception, RunProfile profile
      ) implements InstrumentedOutcome {}
      return new HardInstrumentedOutcome(outputs, exception, profile);
    }

    static InstrumentedOutcome from(List<Object> outputs, RunProfile profile) {
      return from(outputs, null, profile);
    }

    static InstrumentedOutcome from(ProgramExecutionException exception, RunProfile profile) {
      return from(null, exception, profile);
    }

    static InstrumentedOutcome from(ProgramExecutionException exception) {
      return from(null, exception, new RunProfile(List.of()));
    }
  }

  InstrumentedOutcome runInstrumented(List<Object> inputs);

  @Override
  default List<Object> run(List<Object> inputs) throws ProgramExecutionException {
    InstrumentedOutcome instrumentedOutcome = runInstrumented(inputs);
    if (instrumentedOutcome.hasException()) {
      throw instrumentedOutcome.exception();
    }
    return instrumentedOutcome.outputs();
  }

  @Override
  default Outcome safelyRun(List<Object> inputs) {
    return runInstrumented(inputs);
  }
}
