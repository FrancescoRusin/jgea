
package io.github.ericmedvet.jgea.telegram;

import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.request.*;
import com.pengrad.telegrambot.response.GetChatMemberCountResponse;
import com.pengrad.telegrambot.response.GetChatResponse;
import com.pengrad.telegrambot.response.SendResponse;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Set;
import java.util.logging.Logger;

public class TelegramClient {

  protected static final Logger L = Logger.getLogger(TelegramClient.class.getName());
  protected static final Set<String> VIDEO_FILE_EXTENSIONS = Set.of("mpg", "avi", "mp4");
  protected final long chatId;
  protected TelegramBot bot;

  public TelegramClient(String botToken, long chatId) {
    this.chatId = chatId;
    try {
      bot = new TelegramBot(botToken);
    } catch (RuntimeException e) {
      L.severe(String.format("Cannot create bot: %s", e));
    }
  }

  public String getChatInfo() {
    GetChatResponse chatResponse = bot.execute(new GetChat(chatId));
    GetChatMemberCountResponse chatMemberCountResponse = bot.execute(new GetChatMemberCount(chatId));
    String title = chatResponse.chat().title();
    if (title == null) {
      title = chatResponse.chat().firstName() + " " + chatResponse.chat().lastName();
    }
    return "%s (%d members)".formatted(
        title,
        chatMemberCountResponse.count()
    );
  }

  public void sendDocument(File file) {
    try {
      SendResponse response = bot.execute(new SendDocument(chatId, file));
      if (!response.isOk()) {
        L.warning(String.format("Response is not ok: %s", response));
      }
    } catch (Throwable t) {
      L.warning(String.format("Cannot send document: %s", t));
    }
  }

  public void sendImage(BufferedImage image) {
    try {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      ImageIO.write(image, "png", baos);
      baos.close();
      SendResponse response = bot.execute(new SendPhoto(chatId, baos.toByteArray()));
      if (!response.isOk()) {
        L.warning(String.format("Response is not ok: %s", response));
      }
    } catch (Throwable t) {
      L.warning(String.format("Cannot send image: %s", t));
    }
  }

  public void sendText(String string) {
    try {
      SendResponse response = bot.execute(new SendMessage(chatId, string));
      if (!response.isOk()) {
        L.warning(String.format("Response is not ok: %s", response));
      }
    } catch (Throwable t) {
      L.warning(String.format("Cannot send text: %s", t));
    }
  }

  public void sendVideo(File file) {
    try {
      SendResponse response = bot.execute(new SendVideo(chatId, file));
      if (!response.isOk()) {
        L.warning(String.format("Response is not ok: %s", response));
      }
    } catch (Throwable t) {
      L.warning(String.format("Cannot send video: %s", t));
    }
  }

}
