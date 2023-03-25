/**
 * @author "Eric Medvet" on 2022/08/29 for jgea
 */
module io.github.ericmedvet.jgea.problem {
  exports io.github.ericmedvet.jgea.problem.application;
  exports io.github.ericmedvet.jgea.problem.booleanfunction;
  exports io.github.ericmedvet.jgea.problem.classification;
  exports io.github.ericmedvet.jgea.problem.extraction;
  exports io.github.ericmedvet.jgea.problem.extraction.string;
  exports io.github.ericmedvet.jgea.problem.image;
  exports io.github.ericmedvet.jgea.problem.mapper;
  exports io.github.ericmedvet.jgea.problem.symbolicregression;
  exports io.github.ericmedvet.jgea.problem.synthetic;
  requires io.github.ericmedvet.jgea.core;
  requires commons.math3;
  requires java.desktop;
  requires com.google.common;
}