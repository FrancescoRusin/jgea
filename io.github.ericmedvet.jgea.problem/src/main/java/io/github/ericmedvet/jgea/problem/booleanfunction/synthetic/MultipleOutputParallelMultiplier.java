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
package io.github.ericmedvet.jgea.problem.booleanfunction.synthetic;

import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.booleanfunction.BooleanFunction;
import io.github.ericmedvet.jgea.problem.booleanfunction.BooleanUtils;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;
import java.util.List;

public class MultipleOutputParallelMultiplier extends PrecomputedSyntheticBRProblem {
  public MultipleOutputParallelMultiplier(List<Metric> metrics, int n) {
    super(
        BooleanFunction.from(inputs -> compute(inputs, n), 2 * n, 2 * n),
        IndexedProvider.from(
            BooleanUtils.buildCompleteObservations(
                MultivariateRealFunction.varNames("x", 2 * n)
                    .toArray(String[]::new)
            )
        ),
        IndexedProvider.from(
            BooleanUtils.buildCompleteObservations(
                MultivariateRealFunction.varNames("x", 2 * n)
                    .toArray(String[]::new)
            )
        ),
        metrics
    );
  }

  private static boolean[] compute(boolean[] inputs, int n) {
    boolean[] a1 = new boolean[n];
    boolean[] a2 = new boolean[n];
    System.arraycopy(inputs, 0, a1, 0, n);
    System.arraycopy(inputs, n, a2, 0, n);
    int n1 = BooleanUtils.fromBinary(a1);
    int n2 = BooleanUtils.fromBinary(a2);
    return BooleanUtils.toBinary(n1 * n2, 2 * n);
  }
}
