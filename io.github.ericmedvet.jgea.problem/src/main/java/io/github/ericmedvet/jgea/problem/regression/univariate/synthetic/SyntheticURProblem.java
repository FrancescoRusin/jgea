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

import io.github.ericmedvet.jgea.core.problem.TargetEBProblem;
import io.github.ericmedvet.jgea.core.representation.NamedUnivariateRealFunction;
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.regression.univariate.UnivariateRegressionProblem;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public interface SyntheticURProblem extends UnivariateRegressionProblem, TargetEBProblem<NamedUnivariateRealFunction, Map<String, Double>, Double, UnivariateRegressionProblem.Outcome, SequencedMap<String, Double>> {
  @Override
  default String yVarName() {
    return "y";
  }

  static IndexedProvider<Map<String, Double>> tupleProvider(List<double[]> points) {
    List<String> names = MultivariateRealFunction.varNames("x", points.getFirst().length);
    return IndexedProvider.from(
        points.stream()
            .map(
                p -> IntStream.range(0, p.length)
                    .boxed()
                    .collect(
                        Collectors.toMap(
                            names::get,
                            j -> p[j]
                        )
                    )
            )
            .toList()
    );
  }

  static Function<? super Map<String, Double>, ? extends Double> function(ToDoubleFunction<double[]> f, int nOfInputs) {
    return function(f, MultivariateRealFunction.varNames("x", nOfInputs));
  }

  static Function<? super Map<String, Double>, ? extends Double> function(
      ToDoubleFunction<double[]> f,
      List<String> xVarNames
  ) {
    return inputs -> f.applyAsDouble(xVarNames.stream().mapToDouble(inputs::get).toArray());
  }

}
