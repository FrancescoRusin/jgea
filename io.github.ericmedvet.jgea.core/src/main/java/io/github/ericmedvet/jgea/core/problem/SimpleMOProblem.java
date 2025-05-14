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

import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.function.Function;

public interface SimpleMOProblem<S, O> extends MultiObjectiveProblem<S, SequencedMap<String, O>, O> {

  SequencedMap<String, Comparator<O>> comparators();

  static <S, O> SimpleMOProblem<S, O> from(
      SequencedMap<String, Comparator<O>> comparators,
      Function<S, SequencedMap<String, O>> qualityFunction,
      Function<S, SequencedMap<String, O>> validationQualityFunction,
      @SuppressWarnings("OptionalUsedAsFieldOrParameterType") Optional<S> example
  ) {
    record HardSMOProblem<S, O>(
        SequencedMap<String, Comparator<O>> comparators,
        Function<S, SequencedMap<String, O>> qualityFunction,
        Function<S, SequencedMap<String, O>> validationQualityFunction,
        Optional<S> example
    ) implements SimpleMOProblem<S, O> {}
    return new HardSMOProblem<>(comparators, qualityFunction, qualityFunction, example);
  }

  @Override
  default SequencedMap<String, Objective<SequencedMap<String, O>, O>> objectives() {
    return comparators().entrySet()
        .stream()
        .collect(
            Misc.toSequencedMap(
                Map.Entry::getKey,
                e -> new Objective<>(
                    map -> map.get(e.getKey()),
                    e.getValue()
                )
            )
        );
  }

  default SimpleMOProblem<S, O> toReducedSimpleMOProblem(Set<String> objectiveNames) {
    SequencedMap<String, Comparator<O>> reducedComparators = comparators().keySet()
        .stream()
        .filter(objectiveNames::contains)
        .collect(Misc.toSequencedMap(cn -> comparators().get(cn)));
    return from(reducedComparators, qualityFunction(), validationQualityFunction(), example());
  }
}
