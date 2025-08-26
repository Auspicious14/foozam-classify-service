import { Request, Response } from "express";
import * as tf from "@tensorflow/tfjs-node";
import fetch from "node-fetch";

// --- Configuration: URLs for the hosted model and labels ---
const MODEL_URL = "https://raw.githubusercontent.com/Auspicious14/foozam-food-model/main/tfjs_food_model/model.json";
const LABELS_URL = "https://raw.githubusercontent.com/Auspicious14/foozam-food-model/main/food_labels.txt";

// --- Singleton to hold the loaded model and labels ---
let model: tf.LayersModel | null = null;
let labels: string[] | null = null;

async function getOrLoadModelAndLabels() {
  if (model && labels) {
    return { model, labels };
  }

  console.log("Loading model and labels for the first time...");

  // Load the model from the URL
  const loadedModel = await tf.loadLayersModel(MODEL_URL);

  // Fetch and parse the labels
  const response = await fetch(LABELS_URL);
  const text = await response.text();
  const loadedLabels = text.split('\n').filter(label => label.trim() !== ''); // Filter out empty lines

  model = loadedModel;
  labels = loadedLabels;

  console.log(`Model and ${labels.length} labels loaded successfully.`);

  return { model, labels };
}

// --- Prediction Logic ---
async function getTopKClasses(logits: tf.Tensor, topK: number, labels: string[]) {
  const values = await logits.data();
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => b.value - a.value);

  const topClassesAndProbs = [];
  for (let i = 0; i < Math.min(topK, valuesAndIndices.length); i++) {
    const index = valuesAndIndices[i].index;
    topClassesAndProbs.push({
      dish: labels[index] || `Unknown Index: ${index}`,
      confidence: Math.round(valuesAndIndices[i].value * 100),
    });
  }
  return topClassesAndProbs;
}

// --- API Handler ---
export const classifyDish = async (req: Request, res: Response) => {
  try {
    const { model, labels } = await getOrLoadModelAndLabels();

    const imageData = req.body.image;
    if (!imageData) {
      return res.status(400).json({ error: "No image provided" });
    }

    let imageBuffer: Buffer;
    if (typeof imageData === "string") {
      imageBuffer = Buffer.from(
        imageData.replace(/^data:image\/\w+;base64,/, ""),
        "base64"
      );
    } else {
      return res.status(400).json({ error: "Invalid image data" });
    }

    // Preprocess the image
    const imageTensor = tf.tidy(() => {
        const decoded = tf.node.decodeImage(imageBuffer) as tf.Tensor3D;
        const resized = tf.image.resizeBilinear(decoded, [224, 224]);
        const normalized = resized.div(tf.scalar(255.0));
        return normalized.expandDims(0); // Add batch dimension
    });

    const logits = model.predict(imageTensor) as tf.Tensor;
    const predictions = await getTopKClasses(logits, 5, labels);

    tf.dispose([imageTensor, logits]);

    return res.status(200).json({
      confidence: predictions[0]?.confidence || 0,
      predictions: predictions,
    });
  } catch (error) {
    console.error("Classify error:", error);
    // Provide a more specific error message if possible
    const errorMessage = error instanceof Error ? error.message : "Server error";
    return res.status(500).json({ error: errorMessage });
  }
};
