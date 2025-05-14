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
import java.util.Map;
import java.util.SequencedMap;
import java.util.function.Function;

public interface SimpleBBMOProblem<S, B, O> extends BBMOProblem<S, B, SequencedMap<String, O>, O> {

  SequencedMap<String, Objective<B, O>> behaviorObjectives();

  @Override
  default SequencedMap<String, Objective<SequencedMap<String, O>, O>> behaviorQualityObjectives() {
    return behaviorObjectives().entrySet()
        .stream()
        .collect(
            Misc.toSequencedMap(
                Map.Entry::getKey,
                e -> new Objective<>(
                    map -> map.get(e.getKey()),
                    e.getValue().comparator()
                )
            )
        );
  }

  @Override
  default Function<? super B, ? extends SequencedMap<String, O>> behaviorQualityFunction() {
    return b -> behaviorObjectives().entrySet()
        .stream()
        .collect(
            Misc.toSequencedMap(
                Map.Entry::getKey,
                e -> e.getValue().function().apply(b)
            )
        );
  }
}
