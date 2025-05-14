/*-
 * ========================LICENSE_START=================================
 * jgea-core
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
package io.github.ericmedvet.jgea.core.problem;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import java.util.function.Function;

public interface BehaviorBasedProblem<S, B, BQ> extends QualityBasedProblem<S, BehaviorBasedProblem.Outcome<B, BQ>> {
  record Outcome<B, BQ>(B behavior, BQ behaviorQuality) {}

  Function<? super S, ? extends B> behaviorFunction();

  Function<? super B, ? extends BQ> behaviorQualityFunction();

  PartialComparator<BQ> behaviorQualityComparator();

  @Override
  default PartialComparator<Outcome<B, BQ>> qualityComparator() {
    return behaviorQualityComparator().on(Outcome::behaviorQuality);
  }

  @Override
  default Function<S, Outcome<B, BQ>> qualityFunction() {
    return s -> {
      B b = behaviorFunction().apply(s);
      return new Outcome<>(b, behaviorQualityFunction().apply(b));
    };
  }
}
