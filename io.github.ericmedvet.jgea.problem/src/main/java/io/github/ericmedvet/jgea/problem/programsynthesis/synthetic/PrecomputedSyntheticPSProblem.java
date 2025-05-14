/*-
 * ========================LICENSE_START=================================
 * jgea-problem
 * %%
 * Copyright (C) 2018 - 2025 Eric Medvet
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
package io.github.ericmedvet.jgea.problem.programsynthesis.synthetic;

import io.github.ericmedvet.jgea.core.distance.Distance;
import io.github.ericmedvet.jgea.core.problem.PrecomputedTargetEBProblem;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.InstrumentedProgram;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.Program;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.DataFactory;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.programsynthesis.Dissimilarity;
import io.github.ericmedvet.jgea.problem.programsynthesis.ProgramSynthesisProblem;
import java.util.ArrayList;
import java.util.List;
import java.util.SequencedMap;
import java.util.random.RandomGenerator;

public class PrecomputedSyntheticPSProblem extends PrecomputedTargetEBProblem<Program, List<Object>, InstrumentedProgram.InstrumentedOutcome, ProgramSynthesisProblem.Outcome, SequencedMap<String, Double>> implements SyntheticPSProblem {

  private final List<Metric> metrics;
  private final List<Type> inputTypes;
  private final List<Type> outputTypes;
  private final Distance<List<Object>> outputDistance;

  public PrecomputedSyntheticPSProblem(
      Program target,
      IndexedProvider<List<Object>> inputProvider,
      IndexedProvider<List<Object>> validationInputProvider,
      List<Metric> metrics,
      double maxDissimilarity
  ) {
    super(inputs -> ProgramSynthesisProblem.safelyExecute(target, inputs), inputProvider, validationInputProvider);
    this.metrics = metrics;
    inputTypes = target.inputTypes();
    outputTypes = target.outputTypes();
    outputDistance = new Dissimilarity(outputTypes, maxDissimilarity);
  }

  public PrecomputedSyntheticPSProblem(
      Program target,
      List<Metric> metrics,
      double maxDissimilarity,
      DataFactory dataFactory,
      RandomGenerator rnd,
      int nOfCases,
      int nOfValidationCases,
      double maxExceptionRate
  ) {
    this(
        target,
        IndexedProvider.from(buildInputs(nOfCases, maxExceptionRate, target, dataFactory, rnd)),
        IndexedProvider.from(buildInputs(nOfValidationCases, maxExceptionRate, target, dataFactory, rnd)),
        metrics,
        maxDissimilarity
    );
  }

  private static List<List<Object>> buildInputs(
      int n,
      double maxExceptionRate,
      Program program,
      DataFactory dataFactory,
      RandomGenerator rnd
  ) {
    List<List<Object>> cases = new ArrayList<>();
    while (cases.size() < (n * (1d - maxExceptionRate))) {
      List<Object> inputs = program.inputTypes().stream().map(t -> dataFactory.apply(t, rnd)).toList();
      if (!program.safelyRun(inputs).hasException()) {
        cases.add(inputs);
      }
    }
    while (cases.size() < n) {
      cases.add(program.inputTypes().stream().map(t -> dataFactory.apply(t, rnd)).toList());
    }
    return cases;
  }

  @Override
  public List<Metric> metrics() {
    return metrics;
  }

  @Override
  public List<Type> inputTypes() {
    return inputTypes;
  }

  @Override
  public List<Type> outputTypes() {
    return outputTypes;
  }

  @Override
  public Distance<List<Object>> outputDistance() {
    return outputDistance;
  }
}
