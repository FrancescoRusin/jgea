/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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

package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.problem.ExampleBasedProblem;
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.regression.multivariate.MultivariateRegressionProblem;
import io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import java.util.List;
import java.util.Map;

@Discoverable(prefixTemplate = "ea.problem|p.multivariateRegression|mr")
public class MultivariateRegressionProblems {
  private MultivariateRegressionProblems() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static MultivariateRegressionProblem fromData(
      @Param("provider") IndexedProvider<ExampleBasedProblem.Example<Map<String, Double>, Map<String, Double>>> provider,
      @Param(value = "metrics", dSs = {"mse"}) List<UnivariateRegressionProblem.Metric> metrics,
      @Param(value = "nFolds", dI = 10) int nFolds,
      @Param(value = "testFold", dI = 0) int testFold
  ) {
    return MultivariateRegressionProblem.from(
        metrics,
        provider.negatedFold(testFold, nFolds),
        provider.fold(testFold, nFolds)
    );
  }
}
