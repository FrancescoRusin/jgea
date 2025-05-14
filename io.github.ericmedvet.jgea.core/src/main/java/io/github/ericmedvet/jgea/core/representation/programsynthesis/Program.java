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

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.StringParser;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Typed;
import io.github.ericmedvet.jnb.datastructure.NamedFunction;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public interface Program {
  List<Type> inputTypes();

  List<Type> outputTypes();

  List<Object> run(List<Object> inputs) throws ProgramExecutionException;

  interface Outcome {
    List<Object> outputs();

    ProgramExecutionException exception();

    static Outcome from(List<Object> outputs, ProgramExecutionException exception) {
      record HardOutcome(List<Object> outputs, ProgramExecutionException exception) implements Outcome {}
      return new HardOutcome(outputs, exception);
    }

    static Outcome from(List<Object> outputs) {
      return from(outputs, null);
    }

    static Outcome from(ProgramExecutionException exception) {
      return from(null, exception);
    }

    default boolean hasException() {
      return exception() != null;
    }
  }

  default Outcome safelyRun(List<Object> inputs) {
    try {
      return Outcome.from(run(inputs));
    } catch (ProgramExecutionException e) {
      return Outcome.from(e);
    }
  }

  static Program from(
      Function<List<Object>, List<Object>> function,
      List<Type> inputTypes,
      List<Type> outputTypes
  ) {
    record HardProgram(
        Function<List<Object>, List<Object>> function,
        List<Type> inputTypes,
        List<Type> outputTypes
    ) implements Program {
      @Override
      public List<Object> run(List<Object> inputs) throws ProgramExecutionException {
        return safelyRunFunction(function, inputs);
      }

      @Override
      public String toString() {
        return "%s(%s)->(%s)".formatted(
            NamedFunction.name(function),
            inputTypes.stream().map(Object::toString).collect(Collectors.joining(",")),
            outputTypes.stream().map(Object::toString).collect(Collectors.joining(","))
        );
      }
    }
    return new HardProgram(function, inputTypes, outputTypes);
  }

  static <I, O> O safelyRunFunction(Function<I, O> f, I input) throws ProgramExecutionException {
    try {
      return f.apply(input);
    } catch (RuntimeException e) {
      if (e.getCause() instanceof ProgramExecutionException pex) {
        throw pex;
      }
      throw new ProgramExecutionException(e);
    }
  }

  static Program from(Method method) {
    if (!Modifier.isStatic(method.getModifiers())) {
      throw new IllegalArgumentException("Method %s is not static".formatted(method));
    }
    if (!Modifier.isPublic(method.getModifiers())) {
      throw new IllegalArgumentException("Method %s is not public".formatted(method));
    }
    Typed outTyped = method.getAnnotation(Typed.class);
    if (outTyped == null) {
      throw new IllegalArgumentException("Method return is not annotated");
    }
    List<Parameter> parameters = Arrays.stream(method.getParameters()).toList();
    if (parameters.stream().anyMatch(p -> p.getAnnotation(Typed.class) == null)) {
      throw new IllegalArgumentException(
          "Parameter %s is not annotated".formatted(
              parameters.stream().filter(p -> p.getAnnotation(Typed.class) == null).findFirst().orElseThrow()
          )
      );
    }
    List<Type> inputTypes = parameters.stream()
        .map(p -> StringParser.parse(p.getAnnotation(Typed.class).value()))
        .toList();
    Type outputType = StringParser.parse(outTyped.value());
    return Program.from(
        NamedFunction.from(
            inputs -> {
              try {
                Object output = method.invoke(null, inputs.toArray());
                if (!outputType.matches(output)) {
                  throw new IllegalArgumentException(
                      "Wrong output: %s expected, %s produced".formatted(
                          outputType,
                          output.getClass().getSimpleName()
                      )
                  );
                }
                return List.of(output);
              } catch (IllegalAccessException | InvocationTargetException e) {
                throw new RuntimeException(new ProgramExecutionException(e));
              }
            },
            method.getName()
        ),
        inputTypes,
        List.of(outputType)
    );
  }

}
