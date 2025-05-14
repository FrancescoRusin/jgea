/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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
package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import java.util.Comparator;
import java.util.function.Function;

@Discoverable(prefixTemplate = "ea.comparator")
public class Comparators {
  private Comparators() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, T extends Comparable<? super T>> Comparator<X> ascending(
      @Param(value = "of", dNPM = "f.identity()") Function<? super X, ? extends T> beforeF
  ) {
    return Comparator.comparing(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, T extends Comparable<? super T>> Comparator<X> descending(
      @Param(value = "of", dNPM = "f.identity()") Function<? super X, ? extends T> beforeF
  ) {
    //noinspection unchecked
    return (Comparator<X>) Comparator.comparing(beforeF).reversed();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, T extends Comparable<? super T>> PartialComparator<X> pAscending(
      @Param(value = "of", dNPM = "f.identity()") Function<? super X, ? extends T> beforeF
  ) {
    return PartialComparator.from(Comparator.comparing(beforeF));
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, T extends Comparable<? super T>> PartialComparator<X> pDescending(
      @Param(value = "of", dNPM = "f.identity()") Function<? super X, ? extends T> beforeF
  ) {
    return PartialComparator.from(Comparator.comparing(beforeF).reversed());
  }

}
