
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
from collections import namedtuple
from IPython.display import HTML
import rebound
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from shapely.geometry import Polygon

class orbit_diagram:
   """This is a class that defines a two-body orbit using REBOUND, then sets the ideal locations for various annotations,
      such as arrows, labels, and points, that can then be plotted as a static figure or animation.
   """
   def __init__(self, sim, Nframes, n=1, fontsize=20): 
      # Set up the orbit geometry from the initial conditions
      self.sim = sim
      initial_conditions = self.sim.particles[1].calculate_orbit()
      self.e = initial_conditions.e
      self.a = initial_conditions.a
      self.omega = initial_conditions.omega
      self.h = self.sim.particles[1].x * self.sim.particles[1].vy - self.sim.particles[1].y * self.sim.particles[1].vx
      if self.e < 1.0:
         self.b = self.a * np.sqrt(1. - self.e**2) # Semiminor axis of an ellipse
         self.p = self.a * (1. - self.e**2) # Semilatus rectum for an ellipse
      elif self.e > 1.0:
         self.b = -self.a * np.sqrt(self.e**2 - 1.) # Semiminor axis of a hyperbola
         self.p = self.a * (self.e**2 - 1.) # Semilatus rectum for a hyperbola
         self.psi = 2 * np.arcsin(1.0/self.e) # Hyperbolic deflection angle
      else: # Parabolic trajectory
         self.p = self.h**2 / sim.G
         self.a = self.p / 2
         self.b = self.a
      #np.sqrt(self.sim.G * self.a * (1.0 - self.e**2))
      self.Area = np.pi * self.a * self.b

      if self.e != 1.0:
         self.q = self.a * (1. - self.e)
         self.Q = self.a * (1. + self.e)
      else:
         self.q = self.a
      self.fx = self.a * self.e

      # Foci in orbit-centered coordinates
      self.F1 = np.array([self.fx, 0.0])
      self.F2 = np.array([-self.fx, 0.0])

      self.rp = np.array([self.a, 0.0])
      self.ra = np.array([-self.a, 0.0])

      if self.e == 1:
         self.rp += self.F1

      # Compute the orbit 
      self.compute_n_orbits(Nframes, n)

      # Sample the orbit at higher resolution than simulation for better area calculation
      self.sample_fac = 10
      # The following are needed so that the sample points align with the orbit points properly
      Npts = Nframes * self.sample_fac + 1

      if (self.e == 1):
         sim2 = self.sim.copy()
         sim2.particles[1].vx *= 1.0000001
         xyz = np.array(sim2.particles[1].sample_orbit(Npts=Npts,samplingAngle="M",duplicateEndpoint=True)).T
      else:
         xyz = np.array(self.sim.particles[1].sample_orbit(Npts=Npts,samplingAngle="M",duplicateEndpoint=True)).T
      self.xorb = xyz[0]
      self.yorb = xyz[1]
      # Sample point order is reversed from simulation points
      self.xorb = np.flip(self.xorb) 
      self.yorb = np.flip(self.yorb)
      # If the orbit is hyperbolic, create the empty orbit associated with the empty focus
      if self.e > 1.0:
         # Reflect each point in the orbit across the centerline
         lvec = np.array([0.0, 1.0]) # Unit reflection vector
         cpoint = np.array([self.fx * np.cos(self.omega), self.fx * np.sin(self.omega)]) # Reflection origin
         vx = self.xorb + cpoint[0]
         vy = self.yorb + cpoint[1]
         self.xhyp = -vx - cpoint[0]
         self.yhyp = -vy - cpoint[1]

      self.artists  = {
         'Ppoint'       : {'plotcommand'  : 'self.ax.scatter(self.P_x,self.P_y, **self.Ppoint_args, animated=False)',
                           'update_anim' : ['Ppoint.set_offsets([self.P_x, self.P_y])']},
         'Plabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.Plabel_text,xy=(self.Plabel_x,self.Plabel_y),**self.Plabel_args, animated=True)',
                           'update_anim'  : ['Plabel.set_position([self.Plabel_x,self.Plabel_y])', 'Plabel.set_text(self.Plabel_text)']},
         'orbit_path'   : {'plotcommand'  : 'self.ax.plot(self.xorb, self.yorb, **self.orb_path_args)[0]',
                           'update_anim'  : ['orbit_path.set_data(self.xorb, self.yorb)']},
         'hyper_path'   : {'plotcommand'  : 'self.ax.plot(self.xhyp, self.yhyp, **self.hyper_path_args)[0]',
                           'update_anim'  : ['hyper_path.set_data(self.xhyp, self.yhyp)']},
         'rline'        : {'plotcommand'  : 'self.ax.plot(self.rline_x, self.rline_y, **self.rline_args, animated=False)[0]',
                           'update_anim'  : ['rline.set_data([self.rline_x, self.rline_y])']},
         'rarrow'       : {'plotcommand'  : 'self.ax_rot.annotate("",xy=self.rarrowtip,xytext=self.rarrowend, **self.rarrow_args)',  
                           'update_anim'  : ['varrow.set_position(self.rarrowend)', 'rarrow.xy = self.rarrowtip']},
         'varrow'       : {'plotcommand'  : 'self.ax.annotate("",xy=self.varrowtip,xytext=self.varrowend, **self.varrow_args)',  
                           'update_anim'  : ['varrow.set_position(self.varrowend)', 'varrow.xy = self.varrowtip']},
         'vlabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.vlabel_text,xy=(self.vlabel_x,self.vlabel_y), **self.vlabel_args)', 
                           'update_anim'  : ['vlabel.set_position([self.vlabel_x,self.vlabel_y])']},
         'Xref_arrow'   : {'plotcommand'  : 'self.ax.annotate("",xy=self.Xref_arrowtip, xytext=self.Xref_arrowend, **self.ref_arrow_args)',  
                           'update_anim'  : ['Xref_arrow.set_position(self.Xref_arrowend)', 'Xref_arrow.xy = self.Xref_arrowtip']},
         'Xref_label'   : {'plotcommand'  : 'self.ax.annotate(self.Xref_label_text, xy=self.Xref_arrowtip, xytext=(self.Xref_label_x, self.Xref_label_y), **self.ref_label_args)', 
                           'update_anim'  : ['Xref_label.set_position(self.Xref_arrowtip)']},
         'Yref_arrow'   : {'plotcommand'  : 'self.ax.annotate("",xy=self.Yref_arrowtip, xytext=self.Yref_arrowend, **self.ref_arrow_args)',  
                           'update_anim'  : ['Yref_arrow.set_position(self.Yref_arrowend)', 'Yref_arrow.xy = self.Yref_arrowtip']},
         'Yref_label'   : {'plotcommand'  : 'self.ax.annotate(self.Yref_label_text,xy=self.Yref_arrowtip, xytext=(self.Yref_label_x, self.Yref_label_y), **self.ref_label_args)', 
                           'update_anim'  : ['Yref_label.set_position(self.Yref_arrowtip)']},
         'focus1'       : {'plotcommand'  : 'self.ax_rot.scatter(self.F1[0], self.F1[1], **self.focus1_args)',
                           'update_anim'  : ['focus1.set_offsets([self.F1[0], self.F1[1]])']},
         'focus1_label' : {'plotcommand'  : 'self.ax_rot.annotate(self.focus1_label_text,xy=(self.F1[0], self.F1[1]), **self.focus1_label_args)', 
                           'update_anim'  : ['focus1_label.set_position([self.F1[0], self.F1[1]])']},
         'focus2'       : {'plotcommand'  : 'self.ax_rot.scatter(self.F2[0], self.F2[1], **self.focus2_args)',
                           'update_anim'  : ['focus2.set_offsets([self.F2[0], self.F2[1]])']},
         'focus2_label' : {'plotcommand'  : 'self.ax_rot.annotate(self.focus2_label_text,xy=(self.F2[0], self.F2[1]), **self.focus2_label_args)',
                           'update_anim'  : ['focus2_label.set_position([self.F2[0], self.F2[1]])']},
         'peri_point'   : {'plotcommand'  : 'self.ax_rot.scatter(self.rp[0], self.rp[1], **self.peri_args)',
                           'update_anim'  : ['peri_point.set_offsets([self.rp[0], self.rp[1]])']},
         'peri_label'   : {'plotcommand'  : 'self.ax_rot.annotate(self.peri_label_text,xy=(self.rp[0], self.rp[1]), **self.peri_label_args)',
                           'update_anim'  : ['peri_label.set_position([self.rp[0], self.rp[1]])']},
         'apo_point'    : {'plotcommand'  : 'self.ax_rot.scatter(self.ra[0], self.ra[1], **self.apo_args)',
                           'update_anim'  : ['apo_point.set_offsets([self.ra[0], self.ra[1]])']},
         'apo_label'    : {'plotcommand'  : 'self.ax_rot.annotate(self.apo_label_text,xy=(self.ra[0], self.ra[1]), **self.apo_label_args)',
                           'update_anim'  : ['apo_label.set_position([self.ra[0], self.ra[1]])']},
         'centerline'   : {'plotcommand'  : 'self.ax_rot.plot(self.centerline_x, self.centerline_y, **self.centerline_args)[0]',
                           'update_anim'  : ['centerline.set_data(self.centerline_x, self.centerline_y)']},
         'aline'        : {'plotcommand'  : 'self.ax_rot.plot(self.aline_x, self.aline_y, **self.aline_args)[0]',
                           'update_anim'  : ['bline.set_data(self.aline_x, self.aline_y)']},
         'alabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.alabel_text, xy=(self.alabel_x,self.alabel_y), **self.alabel_args)',
                           'update_anim'  : ['blabel.set_position([self.alabel_x, self.alabel_y])']},
         'bline'        : {'plotcommand'  : 'self.ax_rot.plot(self.bline_x, self.bline_y, **self.bline_args)[0]',
                           'update_anim'  : ['bline.set_data(self.bline_x, self.bline_y)']},
         'blabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.blabel_text, xy=(self.blabel_x,self.blabel_y), **self.blabel_args)',
                           'update_anim'  : ['blabel.set_position([self.blabel_x, self.blabel_y])']},
         'impactline'   : {'plotcommand'  : 'self.ax_rot.plot(self.impactline_x, self.impactline_y, **self.impactline_args)[0]',
                           'update_anim'  : ['impactline.set_data(self.impactline_x, self.impactline_y)']},
         'impactlabel'  : {'plotcommand'  : 'self.ax_rot.annotate(self.impactlabel_text, xy=(self.impactlabel_x,self.impactlabel_y), **self.impactlabel_args)',
                           'update_anim'  : ['impactlabel.set_position([self.impactlabel_x, self.impactlabel_y])']},
         'cline'        : {'plotcommand'  : 'self.ax_rot.plot(self.cline_x, self.cline_y, **self.cline_args)[0]',
                           'update_anim'  : ['cline.set_data(self.cline_x, self.cline_y)']},
         'clabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.clabel_text, xy=(self.clabel_x,self.clabel_y), **self.clabel_args)',
                           'update_anim'  : ['clabel.set_position([self.clabel_x, self.clabel_y])']},
         'pline'        : {'plotcommand'  : 'self.ax_rot.plot(self.pline_x, self.pline_y, **self.pline_args)[0]',
                           'update_anim'  : ['pline.set_data(self.pline_x, self.pline_y)']},
         'plabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.plabel_text, xy=(self.plabel_x,self.plabel_y), **self.plabel_args)',
                           'update_anim'  : ['plabel.set_position([self.plabel_x, self.plabel_y])']},
         'flabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.flabel_text,xy=(self.flabel_x,self.flabel_y), **self.flabel_args)', 
                           'update_anim'  : ['flabel.set_position([self.flabel_x, self.flabel_y])', 'flabel.set_text(self.flabel_text)']},
         'farc'         : {'plotcommand'  : 'self.ax_rot.plot(self.f_arc_x, self.f_arc_y, **self.farc_args, animated=False)[0]',
                           'update_anim'  : ['farc.set_data(self.f_arc_x, self.f_arc_y)']},
         'varpi_label'  : {'plotcommand'  : 'self.ax_rot.annotate(self.varpi_label_text,xy=(self.varpi_label_x,self.varpi_label_y), **self.varpi_label_args)', 
                           'update_anim'  : ['varpi_label.set_position([self.varpi_label_x,self.varpi_label_y])']},
         'varpi_arc'    : {'plotcommand'  : 'self.ax_rot.plot(self.varpi_arc_x, self.varpi_arc_y, **self.varpi_arc_args, animated=False)[0]',
                           'update_anim'  : ['varpi_arc.set_data(self.varpi_arc_x, self.varpi_arc_y)']},
         'varpi_arrow'  : {'plotcommand'  : 'self.ax_rot.annotate("",xy=self.varpi_arrowtip,xytext=self.varpi_arrowend, **self.varpi_arrow_args)',
                           'update_anim'  : ['varpi_arrow.set_position(self.varpi_arrowend)', 'varpi_arrow.xy =  self.varpi_arrowtip']},
         'farrow'       : {'plotcommand'  : 'self.ax_rot.annotate("",xy=self.farrowtip,xytext=self.farrowend, **self.farrow_args)',
                           'update_anim'  : ['farrow.set_position(self.farrowend)', 'farrow.xy =  self.farrowtip']},
         'rlabel'       : {'plotcommand'  : 'self.ax_rot.annotate(self.rlabel_text,xy=(self.rlabel_x,self.rlabel_y), **self.rlabel_args)',
                           'update_anim'  : ['rlabel.set_position([self.rlabel_x, self.rlabel_y])']},
         'philabel'     : {'plotcommand'  : 'self.ax_rot.annotate(self.philabel_text,xy=(self.philabel_x,self.philabel_y), **self.philabel_args)', 
                           'update_anim'  : ['philabel.set_position([self.philabel_x,self.philabel_y])', 'philabel.set_text(self.philabel_text)']},
         'phiarc'       : {'plotcommand'  : 'self.ax_rot.plot(self.phi_arc_x, self.phi_arc_y, **self.phiarc_args, animated=False)[0]',
                           'update_anim'  : ['phiarc.set_data(self.phi_arc_x, self.phi_arc_y)']},
         'hline'        : {'plotcommand'  : 'self.ax_rot.plot(self.hline_x, self.hline_y, **self.hline_args, animated=False)[0]',
                           'update_anim'  : ['hline.set_data(self.hline_x, self.hline_y)']},
         'asymptote1'   : {'plotcommand'  : 'self.ax_rot.plot(self.asymptote1_x, self.asymptote1_y, **self.asymptote1_args)[0]',
                           'update_anim'  : ['asymptote1.set_data(self.asymptote1_x, self.asymptote1_y)']},
         'asymptote2'   : {'plotcommand'  : 'self.ax_rot.plot(self.asymptote2_x, self.asymptote2_y, **self.asymptote2_args)[0]',
                           'update_anim'  : ['asymptote2.set_data(self.asymptote2_x, self.asymptote2_y)']},
         'psi_label'    : {'plotcommand'  : 'self.ax_rot.annotate(self.psi_label_text,xy=(self.psi_label_x,self.psi_label_y), **self.psi_label_args)', 
                           'update_anim'  : ['psi_label.set_position([self.psi_label_x,self.psi_label_y])']},
         'psi_arc'      : {'plotcommand'  : 'self.ax_rot.plot(self.psi_arc_x, self.psi_arc_y, **self.psi_arc_args, animated=False)[0]',
                           'update_anim'  : ['psi_arc.set_data(self.psi_arc_x, self.psi_arc_y)']},
         'psi_arrow'    : {'plotcommand'  : 'self.ax_rot.annotate("",xy=self.psi_arrowtip,xytext=self.psi_arrowend, **self.psi_arrow_args)',
                           'update_anim'  : ['psi_arrow.set_position(self.psi_arrowend)', 'psi_arrow.xy =  self.psi_arrowtip']},
      }

      # These are the lists of which artists are static and animated. Only artists in these lists will be plotted
      self.static_list = ['focus1', 'orbit_path']
      self.animated_list = ['Ppoint']
      self.animated_artists = []
      self.static_artists = []

      # Set default values for various annotation quantities that can be overridden
      # Colors and sizes
      self.base_color = 'k'
      self.orbit_color = 'k'
      self.f_color = '#E06015'
      self.a_color = 'k'
      self.b_color = '#377eb8'
      self.c_color = '#DF70AC'
      self.p_color = '#359B73'
      self.v_color = '#C74A47'
      self.phi_color = '#C74A47'
      self.smallcirc = 50.0
      self.largecirc = self.smallcirc * 3
      self.textfont = fontsize

      # Each label has a text value and either a direct xy coordinate or a radial position (relative to focus) and angular offset 
      # (relative to true anomaly if it's animated, or just in polar coordinates otherwise). xy positions are computed 
      # in the update_annotations method if polar values are changed 

      # Set static quantities
      # Orbit path properties
      line_args = {
         'color'      :  self.base_color, 
         'linestyle'  :  '-', 
         'zorder'     :  20,
         'clip_on'    : False,
      }

      arrowprops = {
         'arrowstyle'      : '-|>',
         'mutation_scale'  : 20,
         'connectionstyle' : 'arc3',
         'color'           : self.base_color,
      }

      arrow_args = {
         'xycoords'   : 'data',
         'textcoords' : 'data',
         'arrowprops' : arrowprops,
         'annotation_clip' : False,
         'color'      : self.base_color,
         'zorder'     : 20
      }

      # Shared properties of lines and labels
      label_args = {
         'ha'     : 'center',
         'va'     : 'center',   
         'size'   : self.textfont,
         'color'  : self.base_color,
         'annotation_clip' : False,
         'zorder' : 90,
      }

      # Reference axis arrows
      self.ref_arrow_length = np.abs(self.b * 1.00)
      self.ref_arrowprops = arrowprops.copy()
      self.ref_arrow_args = arrow_args.copy()
      self.ref_arrow_args['arrowprops'] = self.ref_arrowprops
      self.Xref_label_text = '$\hat{X}$'
      self.Yref_label_text = '$\hat{Y}$'
      self.ref_label_args = label_args.copy()
      self.ref_label_args['textcoords'] = 'offset points'
      self.Xref_label_x = -6.0
      self.Xref_label_y = 12.0
      self.Yref_label_x = -12.0
      self.Yref_label_y = -6.0

      self.orb_path_args = line_args.copy()
      self.orb_path_args['color'] = self.orbit_color

      # "Shadow" hyperbolic path on the opposite side of the asymptotes
      self.hyper_path_args = self.orb_path_args.copy()
      self.hyper_path_args['linestyle'] = '--'

      # True anomaly arc (static)
      self.flabel_text = '$f$'
      self.f_arc_rad = 0.20 * self.a # Radius of the arc
      self.flabel_ang_offset = 35.0 * np.pi / 180.0
      self.flabel_rad_offset = self.f_arc_rad + 0.09 * self.a
      self.flabel_args = label_args.copy()
      self.flabel_args['color'] = self.f_color
      self.fline_args = line_args.copy()
      self.fline_args['color'] = self.f_color      

      # Semimajor axis label
      self.alabel_text = '$a$'
      self.alabel_x = -0.4 * self.a 
      self.alabel_y = -0.08

      self.alabel_args = label_args.copy()
      self.alabel_args['color'] = self.a_color

      self.aline_x = [0., -self.a]
      self.aline_y = [0., 0.]
      self.aline_args = line_args.copy()
      self.aline_args['color'] = self.a_color

      # Semiminor axis label and line
      if self.e < 1.0:
         self.bline_x = [0., 0.]
      else:
         self.bline_x = [-self.a, -self.a]
      self.bline_y = [0., np.abs(self.b)]
      self.bline_args = line_args.copy()
      self.bline_args['color'] = self.b_color
      self.blabel_text = '$b$'
      self.blabel_x = self.bline_x[0] - 0.08
      self.blabel_y = np.abs(self.b) / 2
      self.blabel_args = label_args.copy()
      self.blabel_args['color'] = self.b_color

      # Impact parameter line
      if self.e > 1.0:
         # Semiminor axis label and line
         self.impactline_x = [self.F1[0], self.F1[0] + self.b * np.sin(np.pi/2+self.psi/2)]
         self.impactline_y = [self.F1[1], self.F1[1] + self.b * np.cos(np.pi/2+self.psi/2)]
         self.impactline_args = line_args.copy()
         self.impactline_args['color'] = self.b_color
         self.impactlabel_text = '$b$'
         self.impactlabel_x = self.impactline_x[0] + 0.5 * (self.impactline_x[1] - self.impactline_x[0]) - 0.02
         self.impactlabel_y = self.impactline_y[0] + 0.5 * (self.impactline_y[1] - self.impactline_y[0]) - 0.12
         self.impactlabel_args = label_args.copy()
         self.impactlabel_args['color'] = self.b_color

      # Focus-to-center label
      if self.e < 1.0:
         self.cline_x = [0., self.fx]
         self.cline_y = [0., 0.]
      else:
         self.cline_x = [0., -self.a]
         self.cline_y = [0., self.b]

      self.cline_args = line_args.copy()
      self.cline_args['color'] = self.c_color
      self.clabel_text = "$c$"
      self.clabel_x = (self.cline_x[1] - self.cline_x[0]) / 2
      self.clabel_y = (self.cline_y[1] - self.cline_y[0]) / 2 - 0.1
      self.clabel_args = label_args.copy()
      self.clabel_args['color'] = self.c_color
      if self.e > 1:
         self.clabel_args['rotation'] = (-np.arctan2(-self.b,-self.a) + self.omega) * 180./np.pi

      self.pline_x = [self.F2[0], self.F2[0]]
      self.pline_y = [self.F2[1], self.F2[1] + np.abs(self.p)]
      self.pline_args = line_args.copy()
      self.pline_args['color'] = self.p_color
      self.plabel_text = "$p$"
      self.plabel_x = self.pline_x[1] + 0.1
      self.plabel_y = (self.pline_y[1] - self.pline_y[0]) / 2 
      self.plabel_args = label_args.copy()
      self.plabel_args['color'] = self.p_color
      #if self.e > 1:
         #self.clabel_args['rotation'] = -np.arctan2(self.b,-self.a) * 180./np.pi

      # Hyperbolic asymptote lines
      imax = np.argmax(self.r)
      self.hyperbolic_range = 1.10 * np.sqrt((self.x[imax])**2 + (self.y[imax])**2)
      self.asymptote1_x = np.array([-self.hyperbolic_range, self.hyperbolic_range ])
      self.asymptote1_y = self.asymptote1_x * (self.b / self.a)
      self.asymptote2_x = self.asymptote1_x
      self.asymptote2_y = -self.asymptote1_x * (self.b / self.a)
      self.asymptote1_args = line_args.copy()
      self.asymptote1_args['linestyle'] = ':'
      self.asymptote2_args = line_args.copy()
      self.asymptote2_args['linestyle'] = ':'
   
      # Focus point style arguments
      self.focus1_args = {
         'c'      :  self.base_color, 
         's'      :  self.largecirc, 
         'clip_on'    : False,
      } 
      self.focus2_args = self.focus1_args.copy()
      self.focus2_args['c'] = 'w'
      self.focus2_args['edgecolor'] = self.base_color

      # Focus label style
      self.focus1_label_text = '$F$'
      self.focus2_label_text = "$F'$"
      self.focus1_label_args = {
         'xytext'     : (0., -12), 
         'textcoords' : 'offset points',
         'ha'         : 'center',
         'va'         : 'top',
         'size'       : self.textfont,
      }
      self.focus2_label_args = self.focus1_label_args.copy()

      # Peri/apo point
      self.peri_args = {
         'c'       : 'w',
         's'       : self.smallcirc, 
         'clip_on' : False,
      } 

      self.peri_args['edgecolor'] = self.base_color
      self.peri_label_args = {
         'xytext'     : (12., 0.), 
         'textcoords' : 'offset points',
         'ha'         : 'left',
         'va'         : 'center',
         'size'       : self.textfont,
      }
      self.peri_label_text = "$r_p$"
      self.apo_args = self.peri_args.copy()
      self.apo_label_args = {
         'xytext'     : (-12., 0.), 
         'textcoords' : 'offset points',
         'ha'         : 'right',
         'va'         : 'center',
         'size'       : self.textfont,
      }
      
      self.apo_label_text = "$r_a$"

      # Centerline
      if self.e < 1.0:
         self.centerline_x = [-self.a, self.a]
      else:
         self.centerline_x = [self.F1[0], self.F2[0]]
      self.centerline_y = [0., 0.]
      self.centerline_args = line_args.copy()
      self.centerline_args['linestyle'] = '-.'

      # Set animated quantities

      # Position point and label
      self.Plabel_text = '$P$'
      self.Plabel_args = label_args.copy()
      self.Plabel_ang_offset = 4.0 * np.pi / 180.0 
      self.Plabel_rad_offset = 0.08 
      self.Ppoint_args = {
         'c'      : self.orbit_color, 
         's'      : self.smallcirc, 
         'clip_on'    : False,
      }

      # Orbit radius (animated)
      self.rlabel_text = '$r$'
      self.rlabel_args = label_args.copy()
      self.rlabel_args['color'] = self.f_color
      self.rlabel_rad_offset = 0.7 * self.q
      self.rlabel_ang_offset =  15.0 * np.pi / 180.0
      self.rline_args = line_args.copy()
      self.rline_args['color'] = self.f_color

      self.rarrowprops = arrowprops.copy()
      self.rarrowprops['color'] = self.f_color
      self.rarrow_args = arrow_args.copy()
      self.rarrow_args['arrowprops'] = self.rarrowprops
      self.rarrow_args['color'] = self.f_color

      # True anomaly arc (animated)
      self.farrowprops = arrowprops.copy()
      self.farrowprops['color'] = self.f_color
      self.farrow_args = arrow_args.copy()
      self.farrow_args['arrowprops'] = self.farrowprops
      self.farrow_args['color'] = self.f_color
      self.farc_args = line_args.copy()
      self.farc_args['color'] = self.f_color

      # Longitude of ascending node arc
      self.varpi_arc_args = line_args.copy()
      self.varpi_label_text = '$\\varpi$'
      self.varpi_label_args = label_args.copy()
      self.varpi_arc_rad = 0.20 # Radius of the arc
      self.varpi_label_rad_offset = self.varpi_arc_rad + 0.1
      self.varpi_arrowprops = arrowprops.copy()
      self.varpi_arrow_args = arrow_args.copy()
      self.varpi_arrow_args['arrowprops'] = self.varpi_arrowprops

      # Hyperbolic deflection angle arc
      if self.e > 1.0:
         self.psi_arc_args = line_args.copy()
         self.psi_label_text = '$\psi$'
         self.psi_label_args = label_args.copy()
         self.psi_arc_rad = 1.00 # Radius of the arc
         self.psi_label_rad_offset = self.psi_arc_rad + 0.1
         self.psi_arrowprops = arrowprops.copy()
         self.psi_arrow_args = arrow_args.copy()
         self.psi_arrow_args['arrowprops'] = self.psi_arrowprops

      # Velocity vector (animated)
      self.v_length = 0.05 # Length of arrow as fraction of velocity
      self.varrowprops = arrowprops.copy()
      self.varrowprops['color'] = self.v_color
      self.varrow_args = arrow_args.copy()
      self.varrow_args['arrowprops'] = self.varrowprops
      self.varrow_args['color'] = self.v_color

      # The flight path angle arc label (animated)
      self.philabel_text = '$\phi$'
      self.philabel_args = label_args.copy()
      self.philabel_args['color'] = self.phi_color
      self.phi_arc_rad = 0.10 # Radius of arc
      self.philabel_ang_offset = 15.0 * np.pi / 180.0
      self.philabel_rad_offset = self.phi_arc_rad + 0.2
      self.phiarc_args = line_args.copy()
      self.phiarc_args['color'] = self.phi_color

      # Horizontal reference lines for flight path angle (animated)
      self.h_length_left = 0.33 * self.a
      self.h_length_right = 0.33 * self.a
      self.hline_args = line_args.copy()
      self.hline_args['linestyle']  = ":"

      # The velocity vector label
      self.vlabel_text = '$v$'
      self.vlabel_args = label_args.copy()
      self.vlabel_args['color'] = self.v_color
      self.vlabel_ang_offset = 25.0 * np.pi / 180.0
      self.vlabel_rad_offset = self.phi_arc_rad

      # Arrange the various elements in zorder
      # Default lines are 20
      # Default labels are 90
      self.centerline_args['zorder'] = 11
      self.aline_args['zorder'] = 12
      self.pline_args['zorder'] = 12
      self.focus2_args['zorder'] = 13
      self.bline_args['zorder'] = 14
      self.cline_args['zorder'] = 14
      self.orb_path_args['zorder'] = 15
      self.focus1_label_args['zorder'] = 100
      self.focus2_label_args['zorder'] = 100
      self.focus1_args['zorder'] = 150
      self.peri_args['zorder'] = 150
      self.apo_args['zorder'] = 150
      self.Ppoint_args['zorder'] = 200

      # Put animations in their initial position
      self.update_annotations(0)

      return

   def compute_n_orbits(self,Nframes, n):
      # orbit_snap is a snapshot of the orbit at a discrete point in time. 

      # Compute the orbital period from the initial conditions
      orbit_snap = self.sim.particles[1].calculate_orbit()
      if orbit_snap.e < 1.0:
         self.Period = orbit_snap.P
      elif orbit_snap.e > 1.0:
         # Calculate time to pericenter passage and double it as a stand in for "period"
         F = 2 * np.arctanh(np.sqrt((self.e - 1.) / (self.e + 1.)) * np.tan(orbit_snap.f / 2.)) 
         tperi = -np.sqrt((-self.a)**3 / (self.sim.G)) * (self.e * np.sinh(F) - F)
         self.Period = 2 * tperi
      else: # Barker's equation for the parabolic trajectory
         D = np.tan(orbit_snap.f / 2)
         tperi = -0.5 * np.sqrt((self.h**2 / self.sim.G)**3 / self.sim.G) * (D + (1. / 3.) * D**3)
         self.Period = 2 * tperi
   
      # Orbital history arrays
      self.t, self.dt  = np.linspace(start=0.0, stop=self.Period * n, num=Nframes, endpoint=True, retstep=True)
      self.t += self.dt
      self.f = np.zeros(Nframes)
      self.M = np.zeros(Nframes)
      self.E = np.zeros(Nframes)
      self.F = np.zeros(Nframes)
      self.x = np.zeros(Nframes)
      self.y = np.zeros(Nframes)
      self.vx = np.zeros(Nframes)
      self.vy = np.zeros(Nframes)
      self.r = np.zeros(Nframes)
      self.vmag = np.zeros(Nframes)
      self.x_rot = np.zeros(Nframes)
      self.y_rot = np.zeros(Nframes)

      # Integrate the orbit for one period
      for i, t in enumerate(self.t):       
         orbit_snap = self.sim.particles[1].calculate_orbit()
         if self.e < 1.0:
            self.f[i] = orbit_snap.f if orbit_snap.f > 0.0 else 2 * np.pi + orbit_snap.f
         else:
            self.f[i] = orbit_snap.f 
         self.M[i] = orbit_snap.M
         self.x[i] = self.sim.particles[1].x 
         self.y[i] = self.sim.particles[1].y 
         self.vx[i] = self.sim.particles[1].vx 
         self.vy[i] = self.sim.particles[1].vy 
         self.r[i] = np.linalg.norm([self.x[i], self.y[i]])
         self.vmag[i] = np.linalg.norm([self.vx[i], self.vy[i]])
         if orbit_snap.e < 0.99999999:
            self.E[i] = np.arctan2(self.y[i] / (self.a * np.sqrt(1. - self.e**2)), self.x[i] / self.a + self.e)
         elif orbit_snap.e > 1.0:
            self.F[i] = 2 * np.arctanh(np.sqrt((self.e - 1.) / (self.e + 1.)) * np.tan(self.f[i] / 2.))
         self.x_rot[i] = self.r[i] * np.cos(self.f[i]) + self.fx
         self.y_rot[i] = self.r[i] * np.sin(self.f[i])
         self.sim.integrate(t,exact_finish_time=1)
      return

   def update_annotations(self,i):
      """ Positions the annotations for a particular orbit snapshot. """
      self.i = i
      # Compute all animated quantities for snapshot i 
      self.flabel_x = self.flabel_rad_offset * np.cos(self.flabel_ang_offset) + self.fx
      self.flabel_y = self.flabel_rad_offset * np.sin(self.flabel_ang_offset)

      self.Xref_arrowend = np.array([0.0, 0.0]) 
      self.Xref_arrowtip = np.array([self.ref_arrow_length, 0.0])
      self.Yref_arrowend = np.array([0.0, 0.0])  
      self.Yref_arrowtip = np.array([0.0, self.ref_arrow_length])

      # Particle position in orbit-centered coordinates
      self.P_x = self.x[i] 
      self.P_y = self.y[i]

      self.V_x = self.vx[i]
      self.V_y = self.vy[i]

      # Particle label position
      Plabel_ang =  self.Plabel_ang_offset + self.f[i]
      self.Plabel_x = self.x_rot[i] + self.Plabel_rad_offset * np.cos(Plabel_ang)
      self.Plabel_y = self.y_rot[i] + self.Plabel_rad_offset * np.sin(Plabel_ang)

      # r label and line position
      rlabel_ang = self.f[i] + self.rlabel_ang_offset
      self.rline_x = [0., self.P_x]
      self.rline_y = [0., self.P_y]
      # The radius vector 
      self.rarrowend = (self.fx, 0.0)
      self.rarrowtip = (self.x_rot[i], self.y_rot[i])
      rlabel_ang = np.arctan2(self.rarrowtip[1]-self.rarrowend[1], self.rarrowtip[0] - self.rarrowend[0]) + self.rlabel_ang_offset
      self.rlabel_x = self.rlabel_rad_offset * np.cos(rlabel_ang) + self.fx
      self.rlabel_y = self.rlabel_rad_offset * np.sin(rlabel_ang)

      # True anomaly arc and arrowhead
      if self.e < 1.0: # Redfine true anomaly to always be positive for elliptical orbits
         f_arc_ang = self.f[i] if self.f[i] > 0.0 else 2 * np.pi + self.f[i]
      else:
         f_arc_ang = self.f[i] 
      f_ang_arr = np.linspace(0.0, f_arc_ang)
      f_rad_arr = np.full_like(f_ang_arr, self.f_arc_rad)
      self.f_arc_x = f_rad_arr * np.cos(f_ang_arr) + self.fx
      self.f_arc_y = f_rad_arr * np.sin(f_ang_arr)
      self.farrowend = (self.f_arc_x[-2], self.f_arc_y[-2])
      self.farrowtip = (self.f_arc_x[-1], self.f_arc_y[-1])
      #self.flabel_text = r'$f = {}$'.format(round(self.f[i] * 180.0 / np.pi, 1))

      # Longitude of pericenter arc and arrowhead
      varpi_arc_ang = self.omega
      varpi_ang_arr = np.linspace(-varpi_arc_ang, 0.)
      varpi_rad_arr = np.full_like(varpi_ang_arr, self.varpi_arc_rad)
      self.varpi_arc_x = varpi_rad_arr * np.cos(varpi_ang_arr) + self.fx
      self.varpi_arc_y = varpi_rad_arr * np.sin(varpi_ang_arr)
      self.varpi_label_x = self.varpi_label_rad_offset * np.cos(-self.omega * 0.5) + self.fx
      self.varpi_label_y = self.varpi_label_rad_offset * np.sin(-self.omega * 0.5)
      self.varpi_arrowend = (self.varpi_arc_x[-2], self.varpi_arc_y[-2])
      self.varpi_arrowtip = (self.varpi_arc_x[-1], self.varpi_arc_y[-1])

      # Hyperbolic deflection angle arc
      if self.e > 1.0:
         psi_arc_ang = self.psi
         psi_ang_arr = np.linspace(np.pi/2-self.psi/2,np.pi/2+self.psi/2)
         psi_rad_arr = np.full_like(psi_ang_arr, self.psi_arc_rad)
         self.psi_arc_x = psi_rad_arr * np.cos(psi_ang_arr) 
         self.psi_arc_y = psi_rad_arr * np.sin(psi_ang_arr)
         self.psi_label_x = 0.0
         self.psi_label_y = self.psi_label_rad_offset 
         self.psi_arrowend = (self.psi_arc_x[-2], self.psi_arc_y[-2])
         self.psi_arrowtip = (self.psi_arc_x[-1], self.psi_arc_y[-1])

      # Flight path angle arc and label
      self.phi = np.arccos(round(self.h / (self.r[i] * self.vmag[i]), 6)) 
      if 0.0 < self.f[i] < np.pi:
         phi_arc_ang = self.f[i] + np.pi/2 - self.phi 
         philabel_rel_ang = self.f[i] + np.pi/2 - self.philabel_ang_offset 
      else:
         phi_arc_ang = self.f[i] + np.pi/2 + self.phi 
         philabel_rel_ang = self.f[i] + np.pi/2 + self.philabel_ang_offset 

      # The position of the flight path angle label
      self.philabel_x = self.philabel_rad_offset * np.cos(philabel_rel_ang) + self.x_rot[i]
      self.philabel_y = self.philabel_rad_offset * np.sin(philabel_rel_ang) + self.y_rot[i]
      #self.philabel_text = r'$f = {}$'.format(round(self.f[i] * 180.0 / np.pi, 1))

      phi_ang_arr = np.linspace(self.f[i] + np.pi/2, phi_arc_ang)
      phi_rad_arr = np.full_like(phi_ang_arr, self.phi_arc_rad)
      # The array that plots the arc positions
      self.phi_arc_x = phi_rad_arr * np.cos(phi_ang_arr) + self.x_rot[i]
      self.phi_arc_y = phi_rad_arr * np.sin(phi_ang_arr) + self.y_rot[i]

      # Horizontal reference line endpoints (left and right side)
      h0_x = self.h_length_left * np.cos(self.f[i] + np.pi/2) + self.x_rot[i] 
      h0_y = self.h_length_left * np.sin(self.f[i] + np.pi/2) + self.y_rot[i]
      h1_x = self.h_length_right * np.cos(self.f[i] - np.pi/2) + self.x_rot[i]
      h1_y = self.h_length_right * np.sin(self.f[i] - np.pi/2) + self.y_rot[i]

      self.hline_x = [h0_x, h1_x]
      self.hline_y = [h0_y, h1_y]

      # The velocity vector and label
      self.varrowend = (self.P_x, self.P_y)
      self.varrowtip = (self.P_x + self.V_x * self.v_length, self.P_y + self.V_y * self.v_length)
      if self.f[i] < np.pi:
         vlabel_rel_ang = phi_arc_ang - self.vlabel_ang_offset 
      else:
         vlabel_rel_ang = phi_arc_ang + self.vlabel_ang_offset
      self.vlabel_x = self.vlabel_rad_offset * np.cos(vlabel_rel_ang) + self.x_rot[i] 
      self.vlabel_y = self.vlabel_rad_offset * np.sin(vlabel_rel_ang) + self.y_rot[i] 

      return

   def update_plot(self, frame):
      self.update_annotations(frame)

      # Remove any old artists
      for item in self.animated_artists:
         if item.axes == self.ax:
            item.remove() 
      for item in self.static_artists:
         if item.axes == self.ax:
            item.remove()

      # These commands actually draw the artists by executing the plot command string
      animated_artist_list = {key: exec_and_return(self.artists[key]['plotcommand'], self) for key in self.animated_list}
      static_artist_list = {key: exec_and_return(self.artists[key]['plotcommand'], self) for key in self.static_list}

      # Convert the dictionary of objects to namedtuples in order for the Matplotlib animation to update properly
      Animated_Artists = namedtuple('Animated_Artists', sorted(animated_artist_list))
      Static_Artists = namedtuple('Static_Artists', sorted(static_artist_list))

      self.animated_artists = Animated_Artists(**animated_artist_list)
      self.static_artists = Static_Artists(**static_artist_list)

      return 

   def make_rotated_fig(self, figx):

      if self.e < 1.0: # Acommodate the centered ellipse dimensions
         xmax = self.a * 1.10
         ymax = self.b * 1.10
         thetac = np.arctan2(ymax,xmax)
      else: # Accomodate the hyperbolic dimensions
         thetac = np.pi/4 # - np.arcsin(1./self.e)
         xmax = self.hyperbolic_range #* np.cos(thetac)
         ymax = self.hyperbolic_range #* np.sin(thetac)

      rcorner = figx * np.sqrt(1. + (ymax / xmax)**2)
      if self.e > 0.001:
         figx = np.abs(rcorner * np.cos(thetac - self.omega))
         figy = np.abs(rcorner * np.sin(thetac + self.omega))
      else:
         figx = rcorner
         figy = figx

      transform = Affine2D().translate(-self.fx,0.).rotate(self.omega)

      grid_helper = floating_axes.GridHelperCurveLinear(transform, extremes=(-xmax, xmax, -ymax, ymax))
      fig = plt.figure(figsize=(figx, figy))
      ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
      fig.add_subplot(ax)
      ax_rot = ax.get_aux_axes(transform)

      #ax.set_xlim([xmin_rot, xmax_rot])
      #ax.set_ylim([ymin_rot, ymax_rot])
      ax.set_aspect('equal')
      ax_rot.set_aspect('equal')

      # Remove tick labels 
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      ax.grid(False)
      ax.patch.set_visible(False)

      # Remove tick labels of reference frame
      ax.get_xaxis().set_visible(False) 
      ax.get_yaxis().set_visible(False)

      # Remove tick labels of rotated frame
      for axisLoc in ['top', 'bottom', 'left', 'right']:
         ax.axis[axisLoc].set_visible(False)
      ax.patch.set_visible(False)
      plt.tight_layout()

      return fig, ax, ax_rot

   def plot(self, figx, frame=0):
      """ Makes a still frame at a particular frame location (defaults to initial conditions)"""

      fig, self.ax, self.ax_rot = self.make_rotated_fig(figx)

      self.update_plot(frame)
      plt.tight_layout()
      plt.show()

      return fig

   def merged_animation_plot(self, figx, framelist):
      """ Merges animation frames to make a static plot with animation elements overlayed on top of each other"""
      fig, self.ax, self.ax_rot = self.make_rotated_fig(figx)
      self.update_plot(framelist[0])
      for i in framelist[1:]:
         self.update_annotations(i)
         animated_artist_list = {key: exec_and_return(self.artists[key]['plotcommand'], self) for key in self.animated_list}

      plt.tight_layout()
      plt.show()
      return fig


   def make_orbit_anim(self, Nframes, pauseframes, figx):

      # First define the three auxiliary functions that we need for the animation
      def init_fig(orbit):
         return orbit.animated_artists

      def update_artists(orbit):
         """Update artists with data from each frame. """
         for key in orbit.animated_list:
            for update_string in orbit.artists[key]['update_anim']:
               exec('orbit.animated_artists.' + update_string, {'self' : orbit, 'orbit' : orbit}) 
         return orbit.animated_artists

      def step_frames(orbit, Nframes, pauseframes):
         for j in range(Nframes + pauseframes):
            i = j if j < Nframes else 0
            orbit.update_annotations(i) 
            yield orbit

      # Use of partials to generate the animation following example from here:
      # https://alexgude.com/blog/matplotlib-blitting-supernova/

      # Generates the animation object and also returns the initial conditions as a figure object
      fig, self.ax, self.ax_rot = self.make_rotated_fig(figx)
      self.update_plot(0)

      init = partial(init_fig, orbit=self)
      #update = partial(update_artists, animated_artists=self.animated_artists)
      step = partial(step_frames, orbit=self, Nframes=Nframes,pauseframes=pauseframes)

      anim = animation.FuncAnimation(
            fig=fig, 
            func=update_artists, 
            init_func=init,
            frames=step,
            interval=30,
            blit=False,
            save_count=Nframes + pauseframes)

      plt.tight_layout()
      plt.show()
      return anim, fig

   def make_sweep_anim(self, Nframes, pauseframes, figx, nsweep):

      Tlabel_rad_pos = 0.95
      self.Plabel_args['ha'] ='left'
      self.Plabel_args['rotation_mode'] = 'anchor' # Aligns text before rotating, otherwise it looks weird
      self.Plabel_rad_offset = 0.08
      self.Plabel_ang_offset = -130 * np.pi / 180.0

      def sweeper(orbit, si, ns):
         """Generates a new filled polygon artist denoting a sweep area for the Kepler's 2nd law animation"""
         sweep_color_cycle = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
         orbit.xsweep = np.concatenate((np.zeros(1),np.flip(orbit.xorb)[si:si+2]))
         orbit.ysweep = np.concatenate((np.zeros(1),np.flip(orbit.yorb)[si:si+2]))
         orbit.sweep_args = [orbit.xsweep,orbit.ysweep,sweep_color_cycle[orbit.sweep_n % len(sweep_color_cycle)]]
         sweep = orbit.ax.fill(*orbit.sweep_args, animated=True, alpha = 0.5, zorder = 5)

         # Calculate centroid of next sweep
         xnext = np.concatenate((np.zeros(1),orbit.xorb[si:si + ns]))
         ynext = np.concatenate((np.zeros(1),orbit.yorb[si:si + ns]))
         coords = np.vstack((xnext, ynext)).T
         poly = Polygon(coords)
         area_label_x, area_label_y = poly.centroid.x, poly.centroid.y
         orbit.area_label_text = f'Area = {0:.2f}'
         area_label = [orbit.ax.text(s=orbit.area_label_text,x=area_label_x,y=area_label_y, ha='center', va='center', fontsize=orbit.textfont, animated=True, zorder=300)]
         if si != 0:
            time_label_text = f'Time = {orbit.t[int(si/orbit.sample_fac)]/orbit.Period:.2f}'
            time_label = orbit.ax_rot.annotate(time_label_text,xy=(orbit.Plabel_x,orbit.Plabel_y),**orbit.Plabel_args)
         orbit.sweep_start = orbit.sweep_n * ns
         orbit.sweep_n += 1
         return orbit, sweep, area_label

      # First define the three auxiliary functions that we need for the animation
      def init_fig(orbit):
         orbit.sweep_n = 0
         orbit, sweep, area_label = sweeper(orbit=orbit,si=0,ns=nsweep*orbit.sample_fac)
         orbit.sweep_artists += area_label + sweep

         return orbit.sweep_artists

      def update_artists(orbit):
         """Update artists with data from each frame. """
         for key in orbit.animated_list:
            for update_string in orbit.artists[key]['update_anim']:
               exec('self.animated_artists.' + update_string, {'self' : orbit}) 

         orbit.sweep_artists[-1].set_xy(orbit.sweep_xy)
         orbit.sweep_artists[-2].set_text(orbit.area_label_text)
         orbit.animated_artists.Plabel.set_rotation(orbit.Plabel_args['rotation'])

         return orbit.sweep_artists

      def step_frames(orbit, Nframes, pauseframes, nsweep):
         for j in range(Nframes + pauseframes):
            i = j if j < Nframes else Nframes - 1

            orbit.update_annotations(i) 
            orbit.Plabel_x = Tlabel_rad_pos * (orbit.x_rot[i] - self.fx) + orbit.Plabel_rad_offset * np.cos(orbit.Plabel_ang_offset + orbit.f[i])  + self.fx
            orbit.Plabel_y = Tlabel_rad_pos * orbit.y_rot[i] + orbit.Plabel_rad_offset * np.sin(orbit.Plabel_ang_offset + orbit.f[i])
            orbit.Plabel_text = f'Time = {orbit.t[i]/orbit.Period:.2f}'
            orbit.Plabel_args['rotation'] = orbit.f[i] * 180.0 / np.pi + 180.0
            #orbit.Plabel_text = 'X'
            si = i * orbit.sample_fac
            ns = nsweep * orbit.sample_fac
            sweep_end = si + 1
            orbit.xsweep = np.concatenate((np.zeros(1),orbit.xorb[orbit.sweep_start:sweep_end]))
            orbit.ysweep = np.concatenate((np.zeros(1),orbit.yorb[orbit.sweep_start:sweep_end]))
            orbit.sweep_xy = np.vstack((orbit.xsweep, orbit.ysweep)).T

            # Update area calculation
            if orbit.xsweep.size > 2: 
               poly = Polygon(orbit.sweep_xy)
               area_val = poly.area / orbit.Area
            else:
               area_val = 0.

            orbit.area_label_text = f'Area = {area_val:.2f}'

            if i > 0 and i < Nframes - 1 and si % ns == 0:
               update_artists(orbit)
               orbit, sweep, area_label = sweeper(orbit,si,ns)
               orbit.sweep_artists += area_label + sweep
            yield orbit

      # Generates the animation object and also returns the initial conditions as a figure object
      fig, self.ax, self.ax_rot = self.make_rotated_fig(figx)
      self.Plabel_x = Tlabel_rad_pos * (self.x_rot[0] - self.fx) + self.Plabel_rad_offset * np.cos(self.Plabel_ang_offset + self.f[0]) + self.fx
      self.Plabel_y = Tlabel_rad_pos * self.y_rot[0] + self.Plabel_rad_offset * np.sin(self.Plabel_ang_offset + self.f[0])
      self.update_plot(0)

      self.sweep_artists = list(self.animated_artists) 

      init = partial(init_fig, orbit=self)
      update = partial(update_artists)
      step = partial(step_frames, orbit=self, Nframes=Nframes, pauseframes=pauseframes, nsweep=nsweep)

      anim = animation.FuncAnimation(
            fig=fig, 
            func=update,
            init_func=init,
            frames=step,
            interval=30,
            blit=False, 
            save_count=Nframes + pauseframes)

      plt.tight_layout()
      plt.show()
      return anim, fig

def exec_and_return(exec_str, self):
   loc= {}
   exec('temp = ' + exec_str, {'self' : self}, loc)
   return loc['temp']

# Make a new child class of the original orbit_diagram class, but override the plotting and animation methods to allow for the secondary orbit object
class orbit_diagram_two_orbits(orbit_diagram):
    def __init__(self, orbit2, sim, Nframes): 
        super().__init__(sim, Nframes)
        self.orbit2 = orbit2
        self.n1 = 0
        self.n2 = 0
        self.Prat = self.Period / self.orbit2.Period
        self.n2_tick = Nframes 
        self.n1_tick = int((Nframes) / self.Prat)
        
    def make_orbit_anim(self, Nframes, pauseframes, figx):
        """Override the animation method to allow for the secondary orbit"""
        
        def period_counters(orbit):
            orbit.p1_label_text = f'{orbit.n1}'
            orbit.p2_label_text = f'{orbit.n2}'
            p1_label_x = orbit.orbit2.Plabel_x + 0.01
            p1_label_y = orbit.orbit2.Plabel_y + 0.5
            p2_label_x = orbit.Plabel_x + 0.01
            p2_label_y = orbit.Plabel_y + 0.5
            
            plabels = [orbit.ax.text(s=orbit.p1_label_text,x=p1_label_x,y=p1_label_y, ha='center', va='center', fontsize=orbit.textfont, animated=True, zorder=300),
                       orbit.ax.text(s=orbit.p2_label_text,x=p2_label_x,y=p2_label_y, ha='center', va='center', fontsize=orbit.textfont, animated=True, zorder=300)]
            return plabels
            
        # First define the three auxiliary functions that we need for the animation
        def init_fig(orbit):
            plabels = period_counters(orbit)
            orbit.combo_artists += plabels
            return orbit.combo_artists

        def update_artists(orbit):
            """Update artists with data from each frame. """
            for key in orbit.animated_list:
                for update_string in orbit.artists[key]['update_anim']:
                    exec('orbit.animated_artists.' + update_string, {'self' : orbit, 'orbit' : orbit}) 
                    
            for key in orbit.orbit2.animated_list:
                for update_string in orbit.orbit2.artists[key]['update_anim']:
                    exec('orbit.orbit2.animated_artists.' + update_string, {'self' : orbit.orbit2, 'orbit' : orbit})
                    
            orbit.combo_artists[-2].set_text(orbit.p1_label_text)
            orbit.combo_artists[-1].set_text(orbit.p2_label_text)
            
            return orbit.combo_artists

        def step_frames(orbit, Nframes, pauseframes):
            for j in range(Nframes + pauseframes):
               i = j if j < Nframes else 0
               orbit.update_annotations(i)
               orbit.orbit2.update_annotations(i)
               if ((i + 1) % orbit.n1_tick == 0):
                  orbit.n1 += 1
               if ((i + 1) % orbit.n2_tick == 0):
                  orbit.n2 += 1
               orbit.p1_label_text = f'{orbit.n1}'
               orbit.p2_label_text = f'{orbit.n2}'
               yield orbit 

        # Generates the animation object and also returns the initial conditions as a figure object
        fig, self.ax, self.ax_rot = self.make_rotated_fig(figx)
        self.orbit2.ax = self.ax
        self.orbit2.ax_rot = self.ax_rot
        self.update_plot(0)
        self.orbit2.update_plot(0)
        self.combo_artists = list(self.animated_artists) + list(self.orbit2.animated_artists)

        init = partial(init_fig, orbit=self)
        update = partial(update_artists)
        step = partial(step_frames, orbit=self, Nframes=Nframes, pauseframes=pauseframes)

        anim = animation.FuncAnimation(
            fig=fig, 
            func=update, 
            init_func=init,
            frames=step,
            interval=30,
            blit=False,
            save_count=Nframes + pauseframes)

        plt.tight_layout()
        plt.show()
        return anim, fig

if __name__ == '__main__':
   from matplotlib import rc
   from IPython.display import HTML
   import rebound

   # Default output of animation will be a javascript applet
   rc('animation', html='jshtml')

   # Make the LaTeX math look like LaTeX
   rc('text', usetex=True)
   rc('text.latex', preamble=r'\usepackage{amsmath}')

   # Basic plot element values
   # Orbit initial conditions
   a = 1.
   e = 0.6
   f = 0.0001 
   omega = 0.0 #np.pi / 4.0

   sim = rebound.Simulation()
   sim.G = 4 * np.pi**2
   sim.add(m=1.)
   sim.add(a=a,e=e,f=f,omega=omega)

   figx = 8.0

   Nframes = 301
   pauseframes = 100
   
   orbit = orbit_diagram(sim, Nframes)
   orbit.static_list = [
      'focus1',      
      'orbit_path',
   ]
   orbit.animated_list = [
      'Ppoint',
      'Plabel',
      'rline',
   ]
   anim, fig = orbit.make_sweep_anim(Nframes, pauseframes, figx, nsweep=75)
   anim
   #animated_artists, static_artists = update_plot(0, fig, ax, orbit, animated_artists, static_artists)
# %%
