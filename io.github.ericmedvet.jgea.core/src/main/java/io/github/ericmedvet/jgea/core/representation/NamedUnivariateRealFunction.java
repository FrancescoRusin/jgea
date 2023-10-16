
package io.github.ericmedvet.jgea.core.representation;

import io.github.ericmedvet.jsdynsym.core.numerical.UnivariateRealFunction;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
public interface NamedUnivariateRealFunction extends NamedMultivariateRealFunction, UnivariateRealFunction {
  double computeAsDouble(Map<String, Double> input);

  String yVarName();

  static NamedUnivariateRealFunction from(NamedMultivariateRealFunction nmrf) {

    return new NamedUnivariateRealFunction() {
      @Override
      public double computeAsDouble(Map<String, Double> input) {
        return nmrf.compute(input).get(yVarName());
      }

      @Override
      public String yVarName() {
        return nmrf.yVarNames().get(0);
      }

      @Override
      public String toString() {
        return nmrf.toString();
      }

      @Override
      public List<String> xVarNames() {
        return nmrf.xVarNames();
      }
    };
  }

  static NamedUnivariateRealFunction from(
      UnivariateRealFunction urf,
      List<String> xVarNames,
      String yVarName
  ) {
    return new ComposedNamedUnivariateRealFunction(urf, xVarNames, yVarName);
  }

  @Override
  default double applyAsDouble(double[] input) {
    return compute(input)[0];
  }

  @Override
  default Map<String, Double> compute(Map<String, Double> input) {
    return Map.of(yVarName(), computeAsDouble(input));
  }

  @Override
  default List<String> yVarNames() {
    return List.of(yVarName());
  }

  @Override
  default double[] compute(double... xs) {
    if (xs.length != xVarNames().size()) {
      throw new IllegalArgumentException("Wrong number of inputs: %d expected, %d found".formatted(
          xVarNames().size(),
          xs.length
      ));
    }
    Map<String, Double> output = compute(IntStream.range(0, xVarNames().size())
        .mapToObj(i -> Map.entry(xVarNames().get(i), xs[i]))
        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
    return yVarNames().stream().mapToDouble(output::get).toArray();
  }

  @Override
  default int nOfOutputs() {
    return 1;
  }

}
