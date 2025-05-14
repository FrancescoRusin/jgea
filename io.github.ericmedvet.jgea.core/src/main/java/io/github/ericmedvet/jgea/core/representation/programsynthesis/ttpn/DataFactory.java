/*-
 * ========================LICENSE_START=================================
 * jgea-problem
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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Base;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Sequence;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Tuple;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.util.IntRange;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import java.util.List;
import java.util.function.BiFunction;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DataFactory implements BiFunction<Type, RandomGenerator, Object> {
  private final List<Integer> ints;
  private final List<Double> reals;
  private final List<String> strings;
  private final IntRange intRange;
  private final DoubleRange realRange;
  private final IntRange stringLengthRange;
  private final IntRange sequenceSizeRange;

  public DataFactory(
      List<Integer> ints,
      List<Double> reals,
      List<String> strings,
      IntRange intRange,
      DoubleRange realRange,
      IntRange stringLengthRange,
      IntRange sequenceSizeRange
  ) {
    this.ints = ints;
    this.reals = reals;
    this.strings = strings;
    this.intRange = intRange;
    this.realRange = realRange;
    this.stringLengthRange = stringLengthRange;
    this.sequenceSizeRange = sequenceSizeRange;
  }

  @Override
  public Object apply(Type type, RandomGenerator r) {
    if (type instanceof Base base) {
      return switch (base) {
        case INT -> r.nextBoolean() ? ints.get(r.nextInt(ints.size())) : r.nextInt(intRange.min(), intRange.max());
        case REAL -> r.nextBoolean() ? reals.get(r.nextInt(reals.size())) : r.nextDouble(
            realRange.min(),
            realRange.max()
        );
        case BOOLEAN -> r.nextBoolean();
        case STRING -> {
          if (r.nextBoolean()) {
            yield strings.get(r.nextInt(strings.size()));
          }
          yield IntStream.range(0, r.nextInt(stringLengthRange.min(), stringLengthRange.max()))
              .mapToObj(i -> {
                String s = strings.get(r.nextInt(strings.size()));
                int j = r.nextInt(s.length());
                return s.substring(j, j + 1);
              })
              .collect(Collectors.joining());
        }
      };
    }
    if (type instanceof Sequence(Type t)) {
      return IntStream.range(0, r.nextInt(sequenceSizeRange.min(), sequenceSizeRange.max()))
          .mapToObj(i -> apply(t, r))
          .toList();
    }
    if (type instanceof Tuple(List<Type>types)) {
      return types.stream().map(t -> apply(t, r)).toList();
    }
    throw new IllegalArgumentException("Unsupported type %s".formatted(type));
  }
}
