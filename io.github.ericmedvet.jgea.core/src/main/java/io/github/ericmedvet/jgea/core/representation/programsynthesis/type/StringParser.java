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

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record StringParser(String s) {
  private record Token(int start, int end) {
    public String s(String s) {
      return s.substring(start, end);
    }
  }

  private record Node(Type type, Token token) {}

  public static class ParseException extends Exception {
    private final String s;
    private final int i;

    public ParseException(String s, int i) {
      this.s = s;
      this.i = i;
    }

    public ParseException(String message, String s, int i) {
      super(decorate(message, s, i));
      this.s = s;
      this.i = i;
    }

    public ParseException(String message, Throwable cause, String s, int i) {
      super(decorate(message, s, i), cause);
      this.s = s;
      this.i = i;
    }

    private static String decorate(String message, String s, int i) {
      return "%s @%d '%s'".formatted(message, i, s.substring(i));
    }
  }

  public static Type parse(String s) {
    return new StringParser(s).parse();
  }

  public Type parse() {
    try {
      Node node = parseType(0);
      if (node.token.end != s.length()) {
        throw ex(node.token.end, "Unparsed trailing content");
      }
      return node.type;
    } catch (ParseException e) {
      throw new RuntimeException(e);
    }
  }

  private Node parseType(int i) throws ParseException {
    try {
      return parseBase(i);
    } catch (ParseException e) {
      // ignore
    }
    try {
      return parseGeneric(i);
    } catch (ParseException e) {
      // ignore
    }
    try {
      return parseSequence(i);
    } catch (ParseException e) {
      // ignore
    }
    try {
      return parseTuple(i);
    } catch (ParseException e) {
      // ignore
    }
    throw ex(i, "No valid nodes found");
  }

  private Node parseBase(int i) throws ParseException {
    Token t = find(i, "[A-Z]");
    return Arrays.stream(Base.values())
        .filter(b -> b.toString().equals(t.s(s)))
        .findFirst()
        .map(b -> n(b, t))
        .orElseThrow(() -> ex(i, "'%s' is not a Base type".formatted(t.s(s))));
  }

  private Node parseGeneric(int i) throws ParseException {
    Token t = find(i, "[a-z][A-Za-z1-2]*");
    return n(Generic.of(t.s(s)), t);
  }

  private Node parseSequence(int i) throws ParseException {
    Token oParT = find(i, Pattern.quote(Sequence.STRING_PREFIX));
    Node typeNode = parseType(oParT.end);
    Token cParT = find(typeNode.token.end, Pattern.quote(Sequence.STRING_SUFFIX));
    return n(Composed.sequence(typeNode.type), t(oParT.start, cParT.end));
  }

  private Node parseTuple(int i) throws ParseException {
    List<Node> nodes = new ArrayList<>();
    Token oParT = find(i, Pattern.quote(Tuple.STRING_PREFIX));
    nodes.add(parseType(oParT.end));
    Token lastT = nodes.getLast().token;
    while (true) {
      try {
        lastT = find(lastT.end, Pattern.quote(Tuple.STRING_SEPARATOR));
        nodes.add(parseType(lastT.end));
        lastT = nodes.getLast().token;
      } catch (ParseException e) {
        break;
      }
    }
    if (!lastT.equals(nodes.getLast().token)) {
      throw ex(i, "Invalid last token");
    }
    Token cParT = find(lastT.end, Pattern.quote(Tuple.STRING_SUFFIX));
    return n(Composed.tuple(nodes.stream().map(n -> n.type).toList()), t(oParT.start, cParT.end));
  }

  private Token find(int i, String regex) throws ParseException {
    Matcher m = Pattern.compile(regex).matcher(s);
    if (m.find(i) && m.start() == i) {
      return t(i, m.end());
    }
    throw ex(i, "No valid token for '%s'".formatted(regex));
  }

  private ParseException ex(int i, String msg) {
    return new ParseException(msg, s, i);
  }

  private static Node n(Type type, Token token) {
    return new Node(type, token);
  }

  private static Token t(int start, int end) {
    return new Token(start, end);
  }

}
