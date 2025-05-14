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

package io.github.ericmedvet.jgea.core.order;

import io.github.ericmedvet.jgea.core.util.Sized;
import java.util.*;
import java.util.stream.IntStream;

public interface PartiallyOrderedCollection<T> extends Sized {
  void add(T t);

  PartialComparator<? super T> comparator();

  List<Collection<T>> fronts();

  boolean remove(T t);

  static <T> PartiallyOrderedCollection<T> from() {
    return from(
        List.of(),
        (PartialComparator<? super T>) (i1, i2) -> PartialComparator.PartialComparatorOutcome.NOT_COMPARABLE
    );
  }

  static <T> PartiallyOrderedCollection<T> from(Collection<T> ts, PartialComparator<? super T> comparator) {
    PartiallyOrderedCollection<T> poc = new DAGPartiallyOrderedCollection<>(ts, comparator);
    List<Collection<T>> fronts = poc.fronts();
    return new PartiallyOrderedCollection<>() {
      @Override
      public void add(T t) {
        throw new UnsupportedOperationException();
      }

      @Override
      public PartialComparator<? super T> comparator() {
        return comparator;
      }

      @Override
      public List<Collection<T>> fronts() {
        return fronts;
      }

      @Override
      public boolean remove(T t) {
        throw new UnsupportedOperationException();
      }
    };
  }

  static <T> PartiallyOrderedCollection<T> from(T t) {
    List<Collection<T>> fronts = List.of(List.of(t));
    return new PartiallyOrderedCollection<>() {
      @Override
      public void add(T t) {
        throw new UnsupportedOperationException();
      }


      @Override
      public PartialComparator<? super T> comparator() {
        return (k1, k2) -> PartialComparator.PartialComparatorOutcome.SAME;
      }

      @Override
      public List<Collection<T>> fronts() {
        return fronts;
      }

      @Override
      public boolean remove(T t) {
        throw new UnsupportedOperationException();
      }
    };
  }

  static <T> PartiallyOrderedCollection<T> from(Collection<T> ts, Comparator<? super T> comparator) {
    return from(ts, PartialComparator.from(comparator));
  }

  default Collection<T> all() {
    return fronts().stream().flatMap(Collection::stream).toList();
  }

  default Collection<T> firsts() {
    return fronts().getFirst();
  }

  default Collection<T> lasts() {
    return fronts().getLast();
  }

  default Collection<T> mids() {
    List<Collection<T>> fronts = fronts();
    if (fronts.size() > 2) {
      return IntStream.range(1, fronts.size() - 1).mapToObj(fronts::get).flatMap(Collection::stream).toList();
    }
    return List.of();
  }

  @Override
  default int size() {
    return all().size();
  }
}
