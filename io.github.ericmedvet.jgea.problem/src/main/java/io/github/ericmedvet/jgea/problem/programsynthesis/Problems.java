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
package io.github.ericmedvet.jgea.problem.programsynthesis;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Typed;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Problems {
  private Problems() {
  }

  @Typed("S")
  public static String biLongestString(@Typed("S") String s1, @Typed("S") String s2) {
    return s1.length() >= s2.length() ? s1 : s2;
  }

  @Typed("I")
  public static Integer rIntSum(@Typed("R") Double v1, @Typed("R") Double v2) {
    return (int) (v1 + v2);
  }

  @Typed("I")
  public static Integer iArraySum(@Typed("[I]") List<Integer> is) {
    return is.stream().reduce(Integer::sum).orElse(0);
  }

  @Typed("I")
  public static Integer iBiMax(@Typed("I") Integer v1, @Typed("I") Integer v2) {
    return Math.max(v1, v2);
  }

  @Typed("I")
  public static Integer iTriMax(@Typed("I") Integer v1, @Typed("I") Integer v2, @Typed("I") Integer v3) {
    return Math.max(v1, Math.max(v2, v3));
  }

  @Typed("[R]")
  public static List<Double> vScProduct(@Typed("[R]") List<Double> vs, @Typed("R") Double r) {
    return vs.stream().map(v -> v * r).toList();
  }

  @Typed("[<S,I>]")
  public static List<List<Object>> sLengther(@Typed("[S]") List<String> strings) {
    return strings.stream().map(s -> List.<Object>of(s, s.length())).toList();
  }

  @Typed("S")
  public static String triLongestString(@Typed("S") String s1, @Typed("S") String s2, @Typed("S") String s3) {
    return Stream.of(s1, s2, s3).max(Comparator.comparingInt(String::length)).orElseThrow();
  }

  @Typed("R")
  public static Double vProduct(@Typed("[R]") List<Double> v1, @Typed("[R]") List<Double> v2) {
    if (v1.size() != v2.size()) {
      throw new IllegalArgumentException("Input sizes are different: %d and %d".formatted(v1.size(), v2.size()));
    }
    return IntStream.range(0, v1.size()).mapToDouble(i -> v1.get(i) * v2.get(i)).sum();
  }

}
