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
package io.github.ericmedvet.jgea.problem.regression.univariate.synthetic;

import io.github.ericmedvet.jgea.core.problem.PrecomputedTargetEBProblem;
import io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction;
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import java.util.function.Function;

public class PrecomputedSyntheticURProblem extends PrecomputedTargetEBProblem<NamedUnivariateRealFunction, Map<String, Double>, Double, UnivariateRegressionProblem.Outcome, SequencedMap<String, Double>> implements SyntheticURProblem {

  private final List<Metric> metrics;

  public PrecomputedSyntheticURProblem(
      Function<? super Map<String, Double>, ? extends Double> target,
      IndexedProvider<Map<String, Double>> inputProvider,
      IndexedProvider<Map<String, Double>> validationInputProvider,
      List<Metric> metrics
  ) {
    super(target, inputProvider, validationInputProvider);
    this.metrics = metrics;
  }

  @Override
  public List<Metric> metrics() {
    return metrics;
  }
}
