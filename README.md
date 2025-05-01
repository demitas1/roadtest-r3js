# RoadTest - Three.js

Road shape plotting tool.

## Files

```
.
├── backend
│   ├── main.py
│   └── requirements.txt
├── frontend
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── src
│   │   ├── App.tsx
│   │   ├── components
│   │   │   └── Scene.tsx
│   │   ├── main.tsx
│   │   └── style.css
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── LICENSE.txt
└── README.md
```

## Setup

### Back end

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Front end

```
cd frontend
npm install
```

## Run

### Back end

```
cd backend
source venv/bin/activate

python main.py
```

### Front end

```
cd frontend
npm run dev
```

## License

MIT
