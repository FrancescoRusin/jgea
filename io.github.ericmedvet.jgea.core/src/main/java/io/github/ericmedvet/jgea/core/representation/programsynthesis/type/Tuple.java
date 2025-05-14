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

import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record Tuple(List<Type> types) implements Composed {
  public static final String STRING_PREFIX = "<";
  public static final String STRING_SUFFIX = ">";
  public static final String STRING_SEPARATOR = ",";

  public Tuple(List<Type> types) {
    this.types = Collections.unmodifiableList(types);
  }

  @Override
  public boolean canTakeValuesOf(Type other) {
    if (other instanceof Tuple(List<Type>otherTypes)) {
      if (otherTypes.size() == types.size()) {
        for (int i = 0; i < types.size(); i++) {
          if (!types.get(i).canTakeValuesOf(otherTypes.get(i))) {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  @Override
  public Type concrete(Map<Generic, Type> genericTypeMap) {
    if (!isGeneric()) {
      return this;
    }
    List<Type> concreteTypes = new ArrayList<>(types.size());
    for (Type type : types) {
      concreteTypes.add(type.concrete(genericTypeMap));
    }
    return Composed.tuple(concreteTypes);
  }

  @Override
  public double dissimilarity(Object o1, Object o2) {
    @SuppressWarnings("unchecked") List<Object> os1 = (List<Object>) o1;
    @SuppressWarnings("unchecked") List<Object> os2 = (List<Object>) o2;
    return IntStream.range(0, types.size())
        .mapToDouble(i -> types.get(i).dissimilarity(os1.get(i), os2.get(i)))
        .sum();
  }

  @Override
  public Set<Generic> generics() {
    return types.stream()
        .map(Type::generics)
        .reduce(Misc::union)
        .orElse(Set.of());
  }

  @Override
  public boolean matches(Object o) {
    if (o instanceof List<?> list) {
      if (list.size() == types.size()) {
        for (int i = 0; i < types.size(); i++) {
          if (!types.get(i).matches(list.get(i))) {
            return false;
          }
        }
      }
    }
    return true;
  }

  @Override
  public Map<Generic, Type> resolveGenerics(Type concreteType) throws TypeException {
    // spotless:off
    if (concreteType instanceof Tuple(List<Type> otherTypes)) { // spotless:on
      if (otherTypes.size() != types.size()) {
        throw new TypeException("Inconsistent tuple size: %d != %d".formatted(types.size(), otherTypes.size()));
      }
      List<Map<Generic, Type>> maps = new ArrayList<>(types.size());
      for (int i = 0; i < types.size(); i++) {
        maps.add(types.get(i).resolveGenerics(otherTypes.get(i)));
      }
      Map<Generic, Set<Type>> merged = Misc.merge(maps);
      Optional<Map.Entry<Generic, Set<Type>>> oneWrongEntry = merged.entrySet()
          .stream()
          .filter(e -> e.getValue().size() > 1)
          .findAny();
      if (oneWrongEntry.isPresent()) {
        throw new TypeException(
            "Inconsistent types for %s: %s".formatted(
                oneWrongEntry.get().getKey(),
                oneWrongEntry.get().getValue().stream().map(Object::toString).collect(Collectors.joining(", "))
            )
        );
      }
      return merged.entrySet()
          .stream()
          .collect(
              Collectors.toMap(
                  Map.Entry::getKey,
                  e -> e.getValue().stream().findFirst().orElseThrow()
              )
          );
    }
    throw new TypeException("Wrong concrete type: %s does not match %s".formatted(concreteType, toString()));
  }

  @Override
  public int sizeOf(Object o) {
    if (o instanceof List<?> list) {
      if (list.size() == types.size()) {
        return IntStream.range(0, types.size())
            .map(i -> types.get(i).sizeOf(list.get(i)))
            .sum();
      }
      throw new IllegalArgumentException("Inconsistent tuple size: %d != %d".formatted(types.size(), list.size()));
    }
    throw new IllegalArgumentException("Unsupported object type %s".formatted(o.getClass()));
  }

  @Override
  public String toString() {
    return (STRING_PREFIX + "%s" + STRING_SUFFIX).formatted(
        types.stream()
            .map(Object::toString)
            .collect(Collectors.joining(STRING_SEPARATOR))
    );
  }
}
