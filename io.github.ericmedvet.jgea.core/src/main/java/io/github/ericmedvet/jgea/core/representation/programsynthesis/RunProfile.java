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
package io.github.ericmedvet.jgea.core.representation.programsynthesis;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public record RunProfile(List<State> states) {
  public record State(
      Map<Type, Integer> counts,
      Map<Type, Integer> sizes
  ) {
    public static State from(Map<Type, Collection<Object>> data) {
      return new State(
          Misc.transformValues(data, Collection::size),
          data.entrySet()
              .stream()
              .collect(
                  Collectors.toMap(
                      Map.Entry::getKey,
                      e -> e.getValue().stream().mapToInt(o -> e.getKey().sizeOf(o)).sum()
                  )
              )
      );
    }

    public int count() {
      return counts.values().stream().mapToInt(i -> i).sum();
    }

    public int size() {
      return sizes.values().stream().mapToInt(i -> i).sum();
    }
  }

  public double avgCount() {
    return states.stream().mapToDouble(State::count).average().orElse(0d);
  }

  public double avgSize() {
    return states.stream().mapToDouble(State::size).average().orElse(0d);
  }

  public double totCount() {
    return states.stream().mapToDouble(State::count).sum();
  }

  public double totSize() {
    return states.stream().mapToDouble(State::size).distinct().sum();
  }

  public double maxCount() {
    return states.stream().mapToDouble(State::count).max().orElse(0d);
  }

  public double maxSize() {
    return states.stream().mapToDouble(State::size).max().orElse(0d);
  }

  @Override
  public String toString() {
    return "{steps=%d, avg.count=%.1f, avg.size=%.1f}".formatted(states.size(), avgCount(), avgSize());
  }
}
