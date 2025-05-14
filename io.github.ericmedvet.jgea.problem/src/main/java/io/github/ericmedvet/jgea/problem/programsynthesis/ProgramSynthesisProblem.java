/*-
 * ========================LICENSE_START=================================
 * jgea-problem
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
package io.github.ericmedvet.jgea.problem.programsynthesis;

import io.github.ericmedvet.jgea.core.distance.Distance;
import io.github.ericmedvet.jgea.core.problem.SimpleEBMOProblem;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.InstrumentedProgram;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.Program;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ProgramExecutionException;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.RunProfile;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jnb.datastructure.TriFunction;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.SequencedMap;
import java.util.function.BiFunction;

public interface ProgramSynthesisProblem extends SimpleEBMOProblem<Program, List<Object>, InstrumentedProgram.InstrumentedOutcome, ProgramSynthesisProblem.Outcome, Double> {

  double SMOOTH_DISTANCE_THRESHOLD = 0.001d;

  enum Metric implements BiFunction<List<Outcome>, Distance<List<Object>>, Double> {
    SMOOTH_FAIL_RATE(
        (outcomes, d) -> (double) outcomes.stream()
            .filter(outcome -> d.apply(outcome.actual, outcome.executionOutcome.outputs()) > SMOOTH_DISTANCE_THRESHOLD)
            .count() / (double) outcomes.size()
    ), FAIL_RATE(
        (outcomes, d) -> (double) outcomes.stream()
            .filter(outcome -> !Objects.equals(outcome.actual, outcome.executionOutcome.outputs()))
            .count() / (double) outcomes.size()
    ), AVG_RAW_DISSIMILARITY(
        (outcomes, d) -> outcomes.stream()
            .mapToDouble(outcome -> d.apply(outcome.actual, outcome.executionOutcome.outputs()))
            .average()
            .orElseThrow()
    ), EXCEPTION_ERROR_RATE(
        (outcomes, d) -> (double) outcomes.stream()
            .filter(outcome -> !Objects.equals(outcome.actual == null, outcome.executionOutcome.hasException()))
            .count() / (double) outcomes.size()
    ), PROFILE_AVG_STEPS(
        (outcomes, d) -> outcomes.stream()
            .mapToDouble(outcome -> (double) outcome.executionOutcome.profile().states().size())
            .average()
            .orElse(Double.NaN)
    ), PROFILE_AVG_TOT_SIZE(
        (outcomes, d) -> outcomes.stream()
            .mapToDouble(outcome -> outcome.executionOutcome.profile().totSize())
            .average()
            .orElse(Double.NaN)
    );

    private final BiFunction<List<Outcome>, Distance<List<Object>>, Double> function;

    Metric(BiFunction<List<Outcome>, Distance<List<Object>>, Double> function) {
      this.function = function;
    }

    @Override
    public Double apply(List<Outcome> ys, Distance<List<Object>> d) {
      return function.apply(ys, d);
    }

    @Override
    public String toString() {
      return name().toLowerCase();
    }
  }

  record Outcome(List<Object> actual, InstrumentedProgram.InstrumentedOutcome executionOutcome) {}

  List<Type> inputTypes();

  List<Metric> metrics();

  Distance<List<Object>> outputDistance();

  List<Type> outputTypes();

  static InstrumentedProgram.InstrumentedOutcome safelyExecute(Program program, List<Object> inputs) {
    if (program instanceof InstrumentedProgram instrumentedProgram) {
      return instrumentedProgram.runInstrumented(inputs);
    }
    try {
      return InstrumentedProgram.InstrumentedOutcome.from(program.run(inputs), new RunProfile(List.of()));
    } catch (ProgramExecutionException e) {
      return InstrumentedProgram.InstrumentedOutcome.from(e);
    }
  }

  @Override
  default SequencedMap<String, Objective<List<Outcome>, Double>> aggregateObjectives() {
    Distance<List<Object>> outputDistance = outputDistance();
    return metrics().stream()
        .collect(
            Misc.toSequencedMap(
                Enum::toString,
                m -> new Objective<>(outcomes -> m.apply(outcomes, outputDistance), Double::compareTo)
            )
        );
  }

  @Override
  default TriFunction<List<Object>, InstrumentedProgram.InstrumentedOutcome, InstrumentedProgram.InstrumentedOutcome, Outcome> errorFunction() {
    return (inputs, actualExecutionOutcome, executionOutcome) -> new Outcome(
        actualExecutionOutcome.outputs(),
        executionOutcome
    );
  }

  @Override
  default BiFunction<Program, List<Object>, InstrumentedProgram.InstrumentedOutcome> predictFunction() {
    return (ProgramSynthesisProblem::safelyExecute);
  }

  @Override
  default Optional<Program> example() {
    return Optional.of(Program.from(inputs -> List.of(), inputTypes(), outputTypes()));
  }
}
