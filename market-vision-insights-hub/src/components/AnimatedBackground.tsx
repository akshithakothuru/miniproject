
import React, { useEffect, useRef } from 'react';

const AnimatedBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    // Initialize canvas size
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Particle system
    const particlesArray: Particle[] = [];
    const numberOfParticles = 100;
    
    class Particle {
      x: number;
      y: number;
      size: number;
      speedX: number;
      speedY: number;
      color: string;
      
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * 0.5 - 0.25;
        
        // Updated colors with more magenta and teal
        const colors = [
          'rgba(213, 70, 237, 0.7)',   // Magenta
          'rgba(139, 92, 246, 0.7)',    // Purple
          'rgba(20, 241, 149, 0.7)',    // Teal
          'rgba(255, 77, 157, 0.7)',    // Pink
        ];
        this.color = colors[Math.floor(Math.random() * colors.length)];
      }
      
      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        
        if (this.x > canvas.width) {
          this.x = 0;
        } else if (this.x < 0) {
          this.x = canvas.width;
        }
        
        if (this.y > canvas.height) {
          this.y = 0;
        } else if (this.y < 0) {
          this.y = canvas.height;
        }
      }
      
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
      }
    }
    
    // Create particles
    for (let i = 0; i < numberOfParticles; i++) {
      particlesArray.push(new Particle());
    }
    
    // Wave effect
    class Wave {
      private points: { x: number; y: number; originX: number; originY: number; }[];
      private color: string;
      private opacity: number;

      constructor(color: string, opacity: number) {
        this.points = [];
        this.color = color;
        this.opacity = opacity;
        this.initPoints();
      }

      initPoints() {
        const numberOfPoints = 6;
        const width = canvas.width + 200;
        
        for (let i = 0; i <= numberOfPoints; i++) {
          const x = (width / numberOfPoints) * i - 100;
          const y = canvas.height / 2 + Math.random() * 50;
          this.points.push({
            x: x,
            y: y,
            originX: x,
            originY: y
          });
        }
      }

      animate(time: number) {
        for (let i = 0; i <= this.points.length - 1; i++) {
          const point = this.points[i];
          // Wave movement
          point.y = point.originY + Math.sin(time / 1000 + i * 0.5) * 15;
        }
      }

      draw() {
        ctx.beginPath();
        ctx.moveTo(this.points[0].x, this.points[0].y);
        
        for (let i = 0; i < this.points.length - 1; i++) {
          const point = this.points[i];
          const nextPoint = this.points[i + 1];
          const xc = (point.x + nextPoint.x) / 2;
          const yc = (point.y + nextPoint.y) / 2;
          ctx.quadraticCurveTo(point.x, point.y, xc, yc);
        }
        
        const lastPoint = this.points[this.points.length - 1];
        ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, lastPoint.x, lastPoint.y);
        
        ctx.lineTo(canvas.width + 100, canvas.height);
        ctx.lineTo(-100, canvas.height);
        ctx.closePath();
        ctx.fillStyle = `rgba(${this.color}, ${this.opacity})`;
        ctx.fill();
      }
    }

    // Create waves with updated colors
    const waves = [
      new Wave('213, 70, 237', 0.05),   // Magenta
      new Wave('139, 92, 246', 0.03),   // Purple
      new Wave('20, 241, 149', 0.04),   // Teal
      new Wave('255, 77, 157', 0.02),   // Pink
    ];
    
    // Animation function with darker background
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Darker gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#000000');   // Pure black at the top
      gradient.addColorStop(1, '#0a0a0a');   // Very dark gray at the bottom
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Update and draw waves
      const time = Date.now();
      waves.forEach(wave => {
        wave.animate(time);
        wave.draw();
      });
      
      // Update and draw particles
      particlesArray.forEach(particle => {
        particle.update();
        particle.draw();
      });
      
      requestAnimationFrame(animate);
    }
    
    animate();
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full -z-10"
      style={{ pointerEvents: 'none' }}
    />
  );
};

export default AnimatedBackground;
