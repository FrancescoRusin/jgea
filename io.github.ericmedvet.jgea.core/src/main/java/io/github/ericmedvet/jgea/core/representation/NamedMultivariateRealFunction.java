package io.github.ericmedvet.jgea.core.representation;

import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public interface NamedMultivariateRealFunction extends MultivariateRealFunction {
  Map<String, Double> compute(Map<String, Double> input);

  List<String> xVarNames();

  List<String> yVarNames();

  static NamedMultivariateRealFunction from(
          MultivariateRealFunction mrf, List<String> xVarNames, List<String> yVarNames) {
    return new ComposedNamedMultivariateRealFunction(mrf, xVarNames, yVarNames);
  }

  default NamedMultivariateRealFunction andThen(NamedMultivariateRealFunction other) {
    if (!new HashSet<>(yVarNames()).containsAll(other.xVarNames())) {
      throw new IllegalArgumentException("Vars mismatch: required as input=%s; produced as output=%s"
              .formatted(other.xVarNames(), yVarNames()));
    }
    NamedMultivariateRealFunction thisNmrf = this;
    return new NamedMultivariateRealFunction() {
      @Override
      public Map<String, Double> compute(Map<String, Double> input) {
        return other.compute(thisNmrf.compute(input));
      }

      @Override
      public List<String> xVarNames() {
        return thisNmrf.xVarNames();
      }

      @Override
      public List<String> yVarNames() {
        return other.yVarNames();
      }

      @Override
      public String toString() {
        return thisNmrf + "[then:%s]".formatted(other);
      }
    };
  }

  @Override
  default double[] compute(double... xs) {
    if (xs.length != xVarNames().size()) {
      throw new IllegalArgumentException("Wrong number of inputs: %d expected, %d found"
              .formatted(xVarNames().size(), xs.length));
    }
    Map<String, Double> output = compute(IntStream.range(0, xVarNames().size())
            .mapToObj(i -> Map.entry(xVarNames().get(i), xs[i]))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
    return yVarNames().stream().mapToDouble(output::get).toArray();
  }

  @Override
  default NamedMultivariateRealFunction andThen(DoubleUnaryOperator f) {
    return andThen(NamedMultivariateRealFunction.from(
            MultivariateRealFunction.from(
                    new Function<>() {
                      @Override
                      public double[] apply(double[] vs) {
                        return Arrays.stream(vs).map(f).toArray();
                      }

                      @Override
                      public String toString() {
                        return "all:%s".formatted(f);
                      }
                    },
                    nOfOutputs(),
                    nOfOutputs()),
            yVarNames(),
            yVarNames()));
  }

  @Override
  default int nOfInputs() {
    return xVarNames().size();
  }

  @Override
  default int nOfOutputs() {
    return yVarNames().size();
  }
}