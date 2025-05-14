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
import java.util.Comparator;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;

public interface TotalOrderQualityBasedBiProblem<S, O, Q> extends QualityBasedBiProblem<S, O, Q> {

  Comparator<Q> totalOrderComparator();

  @Override
  default PartialComparator<Q> qualityComparator() {
    return (q1, q2) -> {
      int outcome = totalOrderComparator().compare(q1, q2);
      if (outcome == 0) {
        return PartialComparator.PartialComparatorOutcome.SAME;
      }
      if (outcome < 0) {
        return PartialComparator.PartialComparatorOutcome.BEFORE;
      }
      return PartialComparator.PartialComparatorOutcome.AFTER;
    };
  }

  static <S, O, Q> TotalOrderQualityBasedBiProblem<S, O, Q> from(
      QualityBasedBiProblem<S, O, Q> qbBiProblem,
      Comparator<Q> comparator
  ) {
    return from(
        qbBiProblem.outcomeFunction(),
        qbBiProblem.firstQualityFunction(),
        qbBiProblem.secondQualityFunction(),
        comparator,
        qbBiProblem.example()
    );
  }

  static <S, O, Q> TotalOrderQualityBasedBiProblem<S, O, Q> from(
      BiFunction<S, S, O> outcomeFunction,
      Function<O, Q> firstQualityFunction,
      Function<O, Q> secondQualityFunction,
      Comparator<Q> totalOrderComparator,
      @SuppressWarnings("OptionalUsedAsFieldOrParameterType") Optional<S> example
  ) {
    record HardTOQBiProblem<S, O, Q>(
        BiFunction<S, S, O> outcomeFunction,
        Function<O, Q> firstQualityFunction,
        Function<O, Q> secondQualityFunction,
        Comparator<Q> totalOrderComparator,
        Optional<S> example
    ) implements TotalOrderQualityBasedBiProblem<S, O, Q> {
    }
    return new HardTOQBiProblem<>(
        outcomeFunction,
        firstQualityFunction,
        secondQualityFunction,
        totalOrderComparator,
        example
    );
  }
}
