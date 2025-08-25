import { Request, Response } from "express";
import * as tf from "@tensorflow/tfjs-node";
import { FOOD_LABELS } from "./food-labels";

const MODEL_URL = "https://tfhub.dev/google/aiy/vision/classifier/food_V1/1";

let model: tf.LayersModel | null = null;
async function getModel() {
  if (!model) {
    console.log("Loading model...");
    model = await tf.loadLayersModel(MODEL_URL, { fromTFHub: true });
    console.log("Model loaded.");
  }
  return model;
}

// Function to find the top K predictions from the output tensor
async function getTopKClasses(logits: tf.Tensor, topK: number) {
  const values = await logits.data();
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    // Check if the label exists, to prevent errors with the mocked list
    const dish = FOOD_LABELS[topkIndices[i]] || `Unknown Index: ${topkIndices[i]}`;
    topClassesAndProbs.push({
      dish,
      confidence: Math.round(topkValues[i] * 100),
    });
  }
  return topClassesAndProbs;
}


export const classifyDish = async (req: Request, res: Response) => {
  try {
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

    const model = await getModel();

    // The AIY food model expects a 224x224 image.
    const imageTensor = tf.node.decodeImage(imageBuffer)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255)) // Normalize to [0, 1]
        .expandDims(); // Add batch dimension

    const logits = model.predict(imageTensor) as tf.Tensor;
    const predictions = await getTopKClasses(logits, 5);

    imageTensor.dispose();
    logits.dispose();

    return res.status(200).json({
      confidence: predictions[0]?.confidence || 0,
      predictions: predictions,
    });
  } catch (error) {
    console.error("Classify error:", error);
    return res.status(500).json({ error: "Server error" });
  }
};
