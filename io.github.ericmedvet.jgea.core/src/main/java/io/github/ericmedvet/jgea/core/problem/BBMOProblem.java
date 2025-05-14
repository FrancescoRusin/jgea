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

import io.github.ericmedvet.jgea.core.order.ParetoDominance;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.SequencedMap;

public interface BBMOProblem<S, B, BQ, O> extends MultiObjectiveProblem<S, BehaviorBasedProblem.Outcome<B, BQ>, O>, BehaviorBasedProblem<S, B, BQ> {
  SequencedMap<String, Objective<BQ, O>> behaviorQualityObjectives();

  @Override
  default PartialComparator<BQ> behaviorQualityComparator() {
    return (bq1, bq2) -> ParetoDominance.compare(
        bq1,
        bq2,
        bq -> behaviorQualityObjectives().values().stream().map(obj -> (O) obj.function().apply(bq)).toList(),
        behaviorQualityObjectives().values().stream().map(Objective::comparator).toList()
    );
  }

  @Override
  default SequencedMap<String, Objective<Outcome<B, BQ>, O>> objectives() {
    return Misc.sequencedTransformValues(
        behaviorQualityObjectives(),
        obj -> new Objective<>(o -> obj.function().apply(o.behaviorQuality()), obj.comparator())
    );
  }

  @Override
  default PartialComparator<Outcome<B, BQ>> qualityComparator() {
    return behaviorQualityComparator().on(Outcome::behaviorQuality);
  }
}
