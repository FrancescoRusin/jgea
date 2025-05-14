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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.type;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

public record Sequence(Type type) implements Composed {
  public static final String STRING_PREFIX = "[";
  public static final String STRING_SUFFIX = "]";

  @Override
  public boolean canTakeValuesOf(Type other) {
    if (other instanceof Sequence(Type otherType)) {
      return type.canTakeValuesOf(otherType);
    }
    return false;
  }

  @Override
  public Type concrete(Map<Generic, Type> genericTypeMap) {
    if (!isGeneric()) {
      return this;
    }
    return Composed.sequence(type.concrete(genericTypeMap));
  }

  @Override
  public double dissimilarity(Object o1, Object o2) {
    @SuppressWarnings("unchecked") List<Object> os1 = (List<Object>) o1;
    @SuppressWarnings("unchecked") List<Object> os2 = (List<Object>) o2;
    double[] ds = IntStream.range(0, Math.min(os1.size(), os2.size()))
        .mapToDouble(i -> type.dissimilarity(os1.get(i), os2.get(i)))
        .toArray();
    double maxD = Arrays.stream(ds).max().orElse(0d);
    if (maxD == 0) {
      maxD = 1d;
    }
    return Arrays.stream(ds).sum() + maxD * Math.abs(os1.size() - os2.size());
  }

  @Override
  public Set<Generic> generics() {
    return type.generics();
  }

  @Override
  public boolean matches(Object o) {
    if (o instanceof List<?> list) {
      return list.stream().allMatch(lO -> type().matches(lO));
    }
    return false;
  }

  @Override
  public Map<Generic, Type> resolveGenerics(Type concreteType) throws TypeException {
    if (concreteType instanceof Sequence(Type otherType)) {
      return type.resolveGenerics(otherType);
    }
    throw new TypeException("Wrong concrete type: %s does not match %s".formatted(concreteType, toString()));
  }

  @Override
  public int sizeOf(Object o) {
    if (o instanceof List<?> list) {
      return list.stream().mapToInt(type::sizeOf).sum();
    }
    throw new IllegalArgumentException("Unsupported object type %s".formatted(o.getClass()));
  }

  @Override
  public String toString() {
    return (STRING_PREFIX + "%s" + STRING_SUFFIX).formatted(type);
  }
}
