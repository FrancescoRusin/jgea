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

package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public interface Gate {
  record Data(List<List<Object>> lines) {
    public static Data empty() {
      return Data.of(List.of());
    }

    public static Data of(List<List<Object>> lines) {
      return new Data(lines);
    }

    public static Data pair(List<Object> tokens1, List<Object> tokens2) {
      return Data.of(List.of(tokens1, tokens2));
    }

    public static Data pairOne(Object token1, Object token2) {
      return pair(List.of(token1), List.of(token2));
    }

    public static Data single(List<Object> tokens) {
      return Data.of(List.of(tokens));
    }

    public static Data singleOne(Object token) {
      return Data.single(List.of(token));
    }

    public List<Object> all(int lineIndex) {
      return lines().get(lineIndex).stream().toList();
    }

    public <T> List<T> all(int lineIndex, Class<T> clazz) {
      return all(lineIndex).stream().map(clazz::cast).toList();
    }

    public Object one(int lineIndex) {
      return lines().get(lineIndex).getFirst();
    }

    public <T> T one(int lineIndex, Class<T> clazz) {
      return clazz.cast(one(lineIndex));
    }

    public List<Object> ones() {
      return lines.stream().map(List::getFirst).toList();
    }

    public <T> List<T> ones(Class<T> clazz) {
      return ones().stream().map(clazz::cast).toList();
    }

    @Override
    public String toString() {
      return IntStream.range(0, lines().size())
          .mapToObj(
              i -> "%d:%s".formatted(
                  i,
                  all(i).stream().map(t -> "o").collect(Collectors.joining())
              )
          )
          .collect(Collectors.joining(","));
    }
  }

  record InputGate(Type type) implements Gate {
    @Override
    public List<Port> inputPorts() {
      return List.of();
    }

    @Override
    public UnaryOperator<Data> operator() {
      return input -> Data.empty();
    }

    @Override
    public List<Type> outputTypes() {
      return List.of(type);
    }

    @Override
    public String toString() {
      return "IN|-->(%s)".formatted(type);
    }

  }

  record OutputGate(Type type) implements Gate {
    @Override
    public List<Port> inputPorts() {
      return List.of(new Port(type, Port.Condition.EXACTLY, 1));
    }

    @Override
    public UnaryOperator<Data> operator() {
      return input -> Data.empty();
    }

    @Override
    public List<Type> outputTypes() {
      return List.of();
    }

    @Override
    public String toString() {
      return "(%s)--|OUT".formatted(type);
    }
  }

  record Port(Type type, Condition condition, int n) {
    enum Condition {
      EXACTLY, AT_LEAST
    }

    public static Port atLeast(Type type, int n) {
      return new Port(type, Condition.AT_LEAST, n);
    }

    public static Port exactly(Type type, int n) {
      return new Port(type, Condition.EXACTLY, n);
    }

    public static Port single(Type type) {
      return exactly(type, 1);
    }

    @Override
    public String toString() {
      if (condition.equals(Condition.EXACTLY) && n == 1) {
        return type.toString();
      }
      return "%s{%d%s}"
          .formatted(
              type,
              n,
              switch (condition) {
                case EXACTLY -> "";
                case AT_LEAST -> "+";
              }
          );
    }
  }

  List<Port> inputPorts();

  UnaryOperator<Data> operator();

  List<Type> outputTypes();

  static InputGate input(Type type) {
    return new InputGate(type);
  }

  static Gate of(
      List<Port> inputPorts,
      List<Type> outputTypes,
      Function<Data, Data> processingFunction
  ) {
    record HardGate(
        List<Port> inputPorts,
        List<Type> outputTypes,
        UnaryOperator<Data> operator
    ) implements Gate {
      @Override
      public String toString() {
        return "(%s)--|%s|-->(%s)"
            .formatted(
                inputPorts().stream().map(Port::toString).collect(Collectors.joining(",")),
                operator,
                outputTypes().stream().map(Object::toString).collect(Collectors.joining(","))
            );
      }
    }
    UnaryOperator<Data> operator = new UnaryOperator<>() {
      @Override
      public Data apply(Data data) {
        return processingFunction.apply(data);
      }

      @Override
      public String toString() {
        return processingFunction.toString();
      }
    };
    return new HardGate(inputPorts, outputTypes, operator);
  }

  static OutputGate output(Type type) {
    return new OutputGate(type);
  }

  default boolean hasGenerics() {
    return inputPorts().stream().anyMatch(p -> p.type.isGeneric()) || outputTypes().stream().anyMatch(Type::isGeneric);
  }
}
