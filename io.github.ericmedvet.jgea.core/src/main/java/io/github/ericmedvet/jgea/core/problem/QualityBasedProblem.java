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
package io.github.ericmedvet.jgea.core.problem;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import java.util.function.Function;

public interface QualityBasedProblem<S, Q> extends Problem<S>, Function<S, Q> {

  PartialComparator<Q> qualityComparator();

  Function<S, Q> qualityFunction();

  default Function<S, Q> validationQualityFunction() {
    return qualityFunction();
  }

  @Override
  default Q apply(S s) {
    return qualityFunction().apply(s);
  }

  @Override
  default PartialComparatorOutcome compare(S s1, S s2) {
    Function<S, Q> qualityFunction = qualityFunction();
    return qualityComparator()
        .compare(qualityFunction.apply(s1), qualityFunction.apply(s2));
  }

}
