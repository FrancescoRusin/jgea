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
import io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem;
import io.github.ericmedvet.jgea.problem.regression.univariate.synthetic.*;
import io.github.ericmedvet.jnb.core.*;
import java.util.List;
import java.util.Map;

@Discoverable(prefixTemplate = "ea.problem|p.univariateRegression|ur")
public class UnivariateRegressionProblems {
  private UnivariateRegressionProblems() {
  }

  @Alias(
      name = "bundled", passThroughParams = {@PassThroughParam(name = "name", type = ParamMap.Type.STRING), @PassThroughParam(name = "xScaling", value = "none", type = ParamMap.Type.STRING), @PassThroughParam(name = "yScaling", value = "none", type = ParamMap.Type.STRING)
      }, value = // spotless:off
      """
          fromData(provider = ea.provider.num.fromBundled(name = $name; xScaling = $xScaling; yScaling = $yScaling))
          """) // spotless:on
  @Cacheable
  public static UnivariateRegressionProblem fromData(
      @Param("provider") IndexedProvider<ExampleBasedProblem.Example<Map<String, Double>, Map<String, Double>>> provider,
      @Param(value = "metrics", dSs = {"mse"}) List<UnivariateRegressionProblem.Metric> metrics,
      @Param(value = "nFolds", dI = 10) int nFolds,
      @Param(value = "testFold", dI = 0) int testFold
  ) {
    String yVarName = provider.first()
        .output()
        .keySet()
        .stream()
        .findFirst()
        .orElseThrow(() -> new RuntimeException("No output y var in datates"));
    return UnivariateRegressionProblem.from(
        metrics,
        yVarName,
        provider.negatedFold(testFold, nFolds)
            .then(e -> new ExampleBasedProblem.Example<>(e.input(), e.output().get(yVarName))),
        provider.fold(testFold, nFolds)
            .then(e -> new ExampleBasedProblem.Example<>(e.input(), e.output().get(yVarName)))
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static SyntheticURProblem synthetic(
      @Param("name") String name,
      @Param(value = "metrics", dSs = {"mse"}) List<UnivariateRegressionProblem.Metric> metrics,
      @Param(value = "seed", dI = 1) int seed
  ) {
    return switch (name) {
      case "keijzer6" -> new Keijzer6(metrics);
      case "nguyen7" -> new Nguyen7(metrics, seed);
      case "pagie1" -> new Pagie1(metrics);
      case "polynomial4" -> new Polynomial4(metrics);
      case "vladislavleva4" -> new Vladislavleva4(metrics, seed);
      case "korns12" -> new Korns12(metrics, seed);
      case "xor" -> new Xor(metrics);
      default -> throw new IllegalArgumentException("Unknown synthetic function: %s".formatted(name));
    };
  }
}
