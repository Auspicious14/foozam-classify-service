import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { classifyDish } from "./classify"; 

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

app.post("/api/foods/classify", classifyDish);

app.get("/", (_, res) => res.send("Classify service running!"));

app.listen(PORT, () => {
  console.log(`Classify service running on port ${PORT}`);
});