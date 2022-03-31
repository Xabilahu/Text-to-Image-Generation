CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT DEFAULT NULL,
  dirname TEXT DEFAULT NULL,
  gif_filename TEXT DEFAULT NULL,
  image_filename TEXT DEFAULT NULL,
  model_name TEXT CHECK( model_name IN ('vqgan-imagenet', 'vqgan-faceshq', 'vqgan-wikiart', 'biggan-deep-128', 'biggan-deep-256', 'biggan-deep-512') ) DEFAULT NULL,
  seed INTEGER DEFAULT NULL,
  step_size REAL DEFAULT NULL,
  update_freq INTEGER DEFAULT NULL,
  optimization_steps INTEGER DEFAULT NULL,
  similarity_factor REAL DEFAULT NULL,
  smoothing_factor REAL DEFAULT NULL,
  truncation_factor REAL DEFAULT NULL,
  imagenet_classes TEXT DEFAULT "",
  optimize_class BOOLEAN CHECK( optimize_class IN (0, 1) ) DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS job_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  insert_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  start_time TIMESTAMP DEFAULT NULL,
  end_time TIMESTAMP DEFAULT NULL,
  status TEXT CHECK( status IN ('PENDING', 'FINISHED', 'PROCESSING', 'ABORTED', 'ERROR') ) NOT NULL DEFAULT 'PENDING',
  process_id INTEGER NOT NULL,
  image_id INTEGER UNIQUE NOT NULL,
  FOREIGN KEY (image_id) REFERENCES images(id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS votes (
  session_id TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  quality INTEGER CHECK( quality IN (1, 2, 3, 4, 5) ) DEFAULT NULL,
  agreement INTEGER CHECK( agreement IN (1, 2, 3, 4, 5) ) DEFAULT NULL,
  PRIMARY KEY (session_id, image_id),
  FOREIGN KEY (image_id) REFERENCES images(id) ON UPDATE CASCADE ON DELETE CASCADE
);


INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("hogwarts school of witchcraft and wizardry", "./res/inferences/vqgan-imagenet", "hogwarts-school-of-witchcraft-and-wizardry.gif", "hogwarts-school-of-witchcraft-and-wizardry.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("dinosaur extinction", "./res/inferences/vqgan-imagenet", "dinosaur-extinction.gif", "dinosaur-extinction.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Tchaikovsky's Swan Lake", "./res/inferences/vqgan-imagenet", "Tchaikovsky's-Swan-Lake.gif", "Tchaikovsky's-Swan-Lake.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the Mona Lisa in the style of Vincent van Gogh", "./res/inferences/vqgan-imagenet", "the-Mona-Lisa-in-the-style-of-Vincent-van-Gogh.gif", "the-Mona-Lisa-in-the-style-of-Vincent-van-Gogh.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("star wars stormtrooper", "./res/inferences/vqgan-imagenet", "star-wars-stormtrooper.gif", "star-wars-stormtrooper.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a girl in a picnic table using a laptop", "./res/inferences/vqgan-imagenet", "a-girl-in-a-picnic-table-using-a-laptop.gif", "a-girl-in-a-picnic-table-using-a-laptop.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("basque country", "./res/inferences/vqgan-imagenet", "basque-country.gif", "basque-country.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Linus Torvalds", "./res/inferences/vqgan-imagenet", "Linus-Torvalds.gif", "Linus-Torvalds.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("batman in gotham city", "./res/inferences/vqgan-imagenet", "batman-in-gotham-city.gif", "batman-in-gotham-city.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("meme", "./res/inferences/vqgan-imagenet", "meme.gif", "meme.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("an interconnected spider web", "./res/inferences/vqgan-imagenet", "an-interconnected-spider-web.gif", "an-interconnected-spider-web.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("laptop", "./res/inferences/vqgan-imagenet", "laptop.gif", "laptop.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("harry potter", "./res/inferences/vqgan-imagenet", "harry-potter.gif", "harry-potter.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("communist propaganda", "./res/inferences/vqgan-imagenet", "communist-propaganda.gif", "communist-propaganda.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the end of the rainbow", "./res/inferences/vqgan-imagenet", "the-end-of-the-rainbow.gif", "the-end-of-the-rainbow.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("cyberpunk samurai trending on artstation crop", "./res/inferences/vqgan-imagenet", "cyberpunk-samurai-trending-on-artstation-crop.gif", "cyberpunk-samurai-trending-on-artstation-crop.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Ada Lovelace", "./res/inferences/vqgan-imagenet", "Ada-Lovelace.gif", "Ada-Lovelace.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("cyberpunk forest trending on artstation", "./res/inferences/vqgan-imagenet", "cyberpunk-forest-trending-on-artstation.gif", "cyberpunk-forest-trending-on-artstation.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a portrait of Will Smith", "./res/inferences/vqgan-imagenet", "a-portrait-of-Will-Smith.gif", "a-portrait-of-Will-Smith.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("dark matter", "./res/inferences/vqgan-imagenet", "dark-matter.gif", "dark-matter.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a burning forest at night", "./res/inferences/vqgan-imagenet", "a-burning-forest-at-night.gif", "a-burning-forest-at-night.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("zendaya", "./res/inferences/vqgan-imagenet", "zendaya.gif", "zendaya.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("nightmare before christmas", "./res/inferences/vqgan-imagenet", "nightmare-before-christmas.gif", "nightmare-before-christmas.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("cyberpunk samurai trending on artstation", "./res/inferences/vqgan-imagenet", "cyberpunk-samurai-trending-on-artstation.gif", "cyberpunk-samurai-trending-on-artstation.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("university of the basque country", "./res/inferences/vqgan-imagenet", "university of the basque country.gif", "university of the basque country.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("spanish surrealistic painting", "./res/inferences/vqgan-imagenet", "spanish-surrealistic-painting.gif", "spanish-surrealistic-painting.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("deep neural networks", "./res/inferences/vqgan-imagenet", "deep-neural-networks.gif", "deep-neural-networks.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("dog vs human", "./res/inferences/vqgan-imagenet", "dog-vs-human.gif", "", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the persistence of memory by Dali", "./res/inferences/vqgan-imagenet", "the-persistence-of-memory-by-Dali.gif", "the-persistence-of-memory-by-Dali.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("tiktok app logo", "./res/inferences/vqgan-imagenet", "tiktok-app-logo.gif", "tiktok-app-logo.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("greek temples in space", "./res/inferences/vqgan-imagenet", "greek-temples-in-space.gif", "", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("nebula", "./res/inferences/vqgan-imagenet", "nebula.gif", "nebula.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("fuck you", "./res/inferences/vqgan-imagenet", "fuck-you.gif", "fuck-you.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("super mario", "./res/inferences/vqgan-imagenet", "super-mario.gif", "super-mario.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a portrait of Leonardo DiCaprio", "./res/inferences/vqgan-imagenet", "a-portrait-of-Leonardo-DiCaprio.gif", "a-portrait-of-Leonardo-DiCaprio.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a smiling baseball player | photorealism", "./res/inferences/vqgan-imagenet", "a-smiling-baseball-player_photorealism.gif", "a-smiling-baseball-player_photorealism.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("melting clocks in the style of Dali", "./res/inferences/vqgan-imagenet", "melting-clocks-in-the-style-of-Dali.gif", "melting-clocks-in-the-style-of-Dali.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("feminist movement", "./res/inferences/vqgan-imagenet", "feminist-movement.gif", "feminist-movement.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the milky way", "./res/inferences/vqgan-imagenet", "the-milky-way.gif", "the-milky-way.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("olivia rodrigo", "./res/inferences/vqgan-imagenet", "olivia-rodrigo.gif", "olivia-rodrigo.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("asian woman", "./res/inferences/vqgan-faceshq", "asian-woman.gif", "asian-woman.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("The old woman has gray hair with a smile", "./res/inferences/vqgan-faceshq", "The-old-woman-has-gray-hair-with-a-smile.gif", "The-old-woman-has-gray-hair-with-a-smile.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("girl with pointy nose", "./res/inferences/vqgan-faceshq", "girl-with-pointy-nose.gif", "girl-with-pointy-nose.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("woman with catty eyes", "./res/inferences/vqgan-faceshq", "woman-with-catty-eyes.gif", "woman-with-catty-eyes.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("sketch of a bearded man with dense eyebrows", "./res/inferences/vqgan-faceshq", "sketch-of-a-bearded-man-with-dense-eyebrows.gif", "sketch-of-a-bearded-man-with-dense-eyebrows.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("a face of a smiling young redhead woman", "./res/inferences/vqgan-faceshq", "a-face-of-a-smiling-young-redhead-woman.gif", "a-face-of-a-smiling-young-redhead-woman.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("woman with blond hair wearing heavy makeup", "./res/inferences/vqgan-faceshq", "woman-with-blond-hair-wearing-heavy-makeup.gif", "woman-with-blond-hair-wearing-heavy-makeup.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("man smoking", "./res/inferences/vqgan-faceshq", "man-smoking.gif", "man-smoking.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("young boy with glasses", "./res/inferences/vqgan-faceshq", "young-boy-with-glasses.gif", "young-boy-with-glasses.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("african man", "./res/inferences/vqgan-faceshq", "african-man.gif", "african-man.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("surfer boy with long hair", "./res/inferences/vqgan-faceshq", "surfer-boy-with-long-hair.gif", "surfer-boy-with-long-hair.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("man with a hat", "./res/inferences/vqgan-faceshq", "man-with-a-hat.gif", "man-with-a-hat.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("old woman with wrinkles", "./res/inferences/vqgan-faceshq", "old-woman-with-wrinkles.gif", "old-woman-with-wrinkles.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("The man has a big nose and brown hair", "./res/inferences/vqgan-faceshq", "The-man-has-a-big-nose-and-brown-hair.gif", "The-man-has-a-big-nose-and-brown-hair.png", "vqgan-faceshq", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("cyberpunk samurai", "./res/inferences/vqgan-wikiart", "cyberpunk-samurai.gif", "cyberpunk-samurai.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("spacecraft landing on the moon", "./res/inferences/vqgan-wikiart", "spacecraft-landing-on-the-moon.gif", "spacecraft-landing-on-the-moon.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("caribean beach on a sunny day", "./res/inferences/vqgan-wikiart", "caribean-beach-on-a-sunny-day.gif", "caribean-beach-on-a-sunny-day.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("spacecraft landing on the moon unreal engine", "./res/inferences/vqgan-wikiart", "spacecraft-landing-on-the-moon-unreal-engine.gif", "spacecraft-landing-on-the-moon-unreal-engine.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("snowed mountains with a glacial valley", "./res/inferences/vqgan-wikiart", "snowed-mountains-with-a-glacial-valley.gif", "snowed-mountains-with-a-glacial-valley.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("samurai trending on artstation", "./res/inferences/vqgan-wikiart", "samurai-trending-on-artstation.gif", "samurai-trending-on-artstation.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("las meninas in modernist style", "./res/inferences/vqgan-wikiart", "las-meninas-in-modernist-style.gif", "las-meninas-in-modernist-style.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("pirate ship battle in the middle of the ocean", "./res/inferences/vqgan-wikiart", "pirate-ship-battle-in-the-middle-of-the-ocean.gif", "pirate-ship-battle-in-the-middle-of-the-ocean.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Guernica in the style of Vincent van Gogh", "./res/inferences/vqgan-wikiart", "Guernica-in-the-style-of-Vincent-van-Gogh.gif", "Guernica-in-the-style-of-Vincent-van-Gogh.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("las meninas", "./res/inferences/vqgan-wikiart", "las-meninas.gif", "las-meninas.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("nightmare landscape | psychedelic", "./res/inferences/vqgan-wikiart", "nightmare-landscape_psychedelic.gif", "nightmare-landscape_psychedelic.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the beginning of the universe unreal engine", "./res/inferences/vqgan-wikiart", "the-beginning-of-the-universe-unreal-engine.gif", "the-beginning-of-the-universe-unreal-engine.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("starry night", "./res/inferences/vqgan-wikiart", "starry-night.gif", "starry-night.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Guernica by Pablo Picasso", "./res/inferences/vqgan-wikiart", "Guernica-by-Pablo-Picasso.gif", "Guernica-by-Pablo-Picasso.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("spacecraft landing on the moon trending on artstation", "./res/inferences/vqgan-wikiart", "spacecraft-landing-on-the-moon-trending-on-artstation.gif", "spacecraft-landing-on-the-moon-trending-on-artstation.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the beginning of the universe trending on artstation", "./res/inferences/vqgan-wikiart", "the-beginning-of-the-universe-trending-on-artstation.gif", "the-beginning-of-the-universe-trending-on-artstation.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("nightmare landscape", "./res/inferences/vqgan-wikiart", "nightmare-landscape.gif", "nightmare-landscape.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("Guernica in the style of Claude Monet", "./res/inferences/vqgan-wikiart", "Guernica-in-the-style-of-Claude-Monet.gif", "Guernica-in-the-style-of-Claude-Monet.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("the beginning of the universe", "./res/inferences/vqgan-wikiart", "the-beginning-of-the-universe.gif", "the-beginning-of-the-universe.png", "vqgan-wikiart", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of a gold garden spider hanging on web", "./res/poster/vqgan-imagenet", "A-photo-of-a-gold-garden-spider-hanging-on-web.gif", "A-photo-of-a-gold-garden-spider-hanging-on-web.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of dog with a cigarette", "./res/poster/vqgan-imagenet", "A-photo-of-dog-with-a-cigarette.gif", "A-photo-of-dog-with-a-cigarette.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of a lemon with hair and face", "./res/poster/vqgan-imagenet", "A-photo-of-a-lemon-with-hair-and-face.gif", "A-photo-of-a-lemon-with-hair-and-face.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of a polar bear under the blue sky", "./res/poster/vqgan-imagenet", "A-photo-of-a-polar-bear-under-the-blue-sky.gif", "A-photo-of-a-polar-bear-under-the-blue-sky.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of a blue bird with big beak", "./res/poster/vqgan-imagenet", "A-photo-of-a-blue-bird-with-big-beak.gif", "A-photo-of-a-blue-bird-with-big-beak.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor) VALUES ("A photo of a flying penguin", "./res/poster/vqgan-imagenet", "A-photo-of-a-flying-penguin.gif", "A-photo-of-a-flying-penguin.png", "vqgan-imagenet", 0.5, 100, 5, 1.0);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of a gold garden spider hanging on web", "./res/poster/biggan-deep-512", "A-photo-of-a-gold-garden-spider-hanging-on-web.gif", "A-photo-of-a-gold-garden-spider-hanging-on-web.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "barn_spider", 1);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of dog with a cigarette", "./res/poster/biggan-deep-512", "A-photo-of-dog-with-a-cigarette.gif", "A-photo-of-dog-with-a-cigarette.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "basenji", 1);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of a lemon with hair and face", "./res/poster/biggan-deep-512", "A-photo-of-a-lemon-with-hair-and-face.gif", "A-photo-of-a-lemon-with-hair-and-face.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "lemon", 1);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of a polar bear under the blue sky", "./res/poster/biggan-deep-512", "A-photo-of-a-polar-bear-under-the-blue-sky.gif", "A-photo-of-a-polar-bear-under-the-blue-sky.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "American_black_bear", 1);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of a blue bird with big beak", "./res/poster/biggan-deep-512", "A-photo-of-a-blue-bird-with-big-beak.gif", "A-photo-of-a-blue-bird-with-big-beak.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "cock", 1);
INSERT INTO images (prompt, dirname, gif_filename, image_filename, model_name, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES ("A photo of a flying penguin", "./res/poster/biggan-deep-512", "A-photo-of-a-flying-penguin.gif", "A-photo-of-a-flying-penguin.png", "biggan-deep-512", 0.5, 100, 5, 10.0, 0.1, 1.0, "king_penguin", 1);
